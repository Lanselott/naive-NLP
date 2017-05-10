import nltk
from nltk.corpus import wordnet
from nltk.wsd import lesk
from nltk.corpus import sentiwordnet as swn
import yaml
from pprint import pprint
import sys
import os
import re

class Splitter(object):
	def __init__(self):
	        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
	        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()
	def split(self, text):
	        sentences = self.nltk_splitter.tokenize(text)
        	tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        	return tokenized_sentences

class POSTagger(object):

    def __init__(self):
        pass

    def pos_tag(self, sentences):
        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        #adapt format
        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos


class TagConverter(object):
	def __init__(self):
		pass
	def convert_tag(self, treebank_tag):
			if treebank_tag.startswith('J'):
				return wordnet.ADJ
			elif treebank_tag.startswith('V'):
				return wordnet.VERB
			elif treebank_tag.startswith('N'):
				return wordnet.NOUN
			elif treebank_tag.startswith('R'):
				return wordnet.ADV
			else:
				return ''

class DictionaryTagger(object):
    def __init__(self, dictionary_paths):
        files = [open(path, 'r') for path in dictionary_paths]
        dictionaries = [yaml.load(dict_file,yaml.BaseLoader) for dict_file in files]
        map(lambda x: x.close(), files)
        self.dictionary = {}
        self.max_key_size = 0
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))

    def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemmas=False):

        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        while (i < N):
            j = min(i + self.max_key_size, N) #avoid overflow
            tagged = False
            while (j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                if tag_with_lemmas:
                    literal = expression_lemma
                else:
                    literal = expression_form
                if literal in self.dictionary:
                    #self.logger.debug("found: %s" % literal)
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence


if __name__ == '__main__':
	text = "Your body is your brain.The blogger used a very concise sentence to summarize his own understanding of embodied cognition.The cognition exists in brain, the brain coordinate with the body, and the body belong to the environment.He used a triangular diagram to show this relationship which also helps me a lot to understand embodied cognitionAlso an example was used in the blog..Angier in Yale University apart 41 students into two groups randomly. Group A students hold the warm coffee and Group B hold the cold one. The result is that, students in group A are more likely to assess the character as an enthusiasm, friendly one.I have heard of this experiment before,in fact environment can really influence our mind and temperature is one of the factors."
	print(text)
	splitter = Splitter()
	postagger = POSTagger()
	tag_converter = TagConverter()
	splitted_sentences = splitter.split(text)
	pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
	dicttagger = DictionaryTagger(['inv.yml'])
	dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)

	#use sentiworldnet
	score = 0
	word_num = 0
	for sentence in dict_tagged_sentences:
		whole_sentence = ''
		for word in sentence:
			whole_sentence = whole_sentence + word[0] + " "
		negative = False
		for word in sentence:
			if word[2][0] == "inv":
				negative = True
			else:
				if tag_converter.convert_tag(word[2][len(word[2]) - 1]) != "":
					synset = lesk(whole_sentence,word[0],tag_converter.convert_tag(word[2][len(word[2]) - 1]))
					if synset is not None:
						if negative:
							score = score + swn.senti_synset(synset.name()).neg_score() - swn.senti_synset(synset.name()).pos_score()
							word_num = word_num + 1
						else:
							score = score + swn.senti_synset(synset.name()).pos_score() - swn.senti_synset(synset.name()).neg_score()
							word_num = word_num + 1
	print(score/word_num)
	content = None
	with open("word_set.txt") as f:
		content = f.readlines()
		content = [x.strip() for x in content]
	#print(content)
	seed_dict = dict()
	for word in content:
		for syn in wordnet.synsets(word):
			if not seed_dict.has_key(syn):
				seed_dict[syn] = swn.senti_synset(syn.name()).pos_score() - swn.senti_synset(syn.name()).neg_score()
	#use own dictionary
	print(seed_dict)
	score = 0
	word_num = 0
	for sentence in dict_tagged_sentences:
		whole_sentence = ''
		for word in sentence:
			whole_sentence = whole_sentence + word[0] + " "
		negative = False
		for word in sentence:
			if word[2][0] == "inv":
				negative = True
			else:
				if tag_converter.convert_tag(word[2][len(word[2]) - 1]) != "":
					synset = lesk(whole_sentence,word[0],tag_converter.convert_tag(word[2][len(word[2]) - 1]))
					if synset is not None:
						if negative:
							if seed_dict.has_key(synset):
								score = score + (- seed_dict[synset])
								word_num = word_num + 1
						else:
							if seed_dict.has_key(synset):
								score = score + seed_dict[synset]
								word_num = word_num + 1
	print(score/word_num)
