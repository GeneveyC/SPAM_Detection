import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk, sys

def count_number(sms):
	sum = sms.count("0")+sms.count("1")+sms.count("2")+sms.count("3")+sms.count("4")+sms.count("5")+sms.count("6")+sms.count("7")+sms.count("8")+sms.count("9")
	return sum

def count_caractere(sms):
	sum = sms.count("!")+sms.count("#")+sms.count("$")+sms.count("%")+sms.count("&")+sms.count("(")+sms.count(")")+sms.count("*")+sms.count("+")+sms.count(",")+sms.count("-")+sms.count("-")+sms.count("/")+sms.count(":")+sms.count(";")+sms.count("=")+sms.count("<")+sms.count(">")+sms.count("?")+sms.count("@")+sms.count("[")+sms.count("]")+sms.count("^")+sms.count("_")+sms.count("|")+sms.count("{")+sms.count("}")+sms.count("~")
	return sum

# Function to extract feature for training
def extract_feature(sms):
	return len(sms),count_number(sms),count_caractere(sms)

if __name__ == '__main__':
	# Create DataFrame with spam.csv
	df = pd.read_csv('./data/spam.csv', encoding='latin-1')
	df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)
	df = df.rename(columns= {'v1':'Class','v2':'Text'})

	# Count the length of sms
	df['Count'] = 0
	for i in np.arange(0,len(df.Text)):
		df.loc[i,'Count'] = len(df.loc[i,'Text'])
	print("Unique values in the Class set: ", df.Class.unique())

	# Count the number of integer
	df['Number'] = 0
	for i in np.arange(0,len(df.Text)):
		df.loc[i,'Number'] = count_number(df.loc[i,'Text'])

	# Count the number of special caracter
	df['Caracter'] = 0
	for i in np.arange(0, len(df.Text)):
		df.loc[i, 'Caracter'] = count_caractere(df.loc[i, 'Text'])

	# Replace "ham" and "spam" by 0,1
	df = df.replace(['ham','spam'],[0,1])
	df.head()

	# Count the number of Ham message
	ham = df[df.Class == 0]
	ham_count = pd.DataFrame(pd.value_counts(ham['Count'], sort=True).sort_index())
	print("Number of ham messages in data set:",ham['Class'].count())
	print("Ham count value", ham_count['Count'].count())

	# Count the number of Spam message
	spam = df[df.Class == 1]
	spam_count = pd.DataFrame(pd.value_counts(spam['Count'], sort=True).sort_index())
	print("Number of spam messages in data set:", spam['Count'].count())
	print("Spam count value:",spam_count['Count'].count())

	# Show SMS Ham by length of message
	ax = plt.axes()
	xline_ham = np.linspace(0, len(ham_count) - 1, len(ham_count))
	ax.bar(xline_ham, ham_count['Count'], width=2.2, color='r')
	ax.set_title('SMS Ham by length of message')
	plt.xlabel('length')
	plt.ylabel('frequency')
	plt.show()

	# Show SMS SPam by length of message
	ax = plt.axes()
	xline_spam = np.linspace(0, len(spam_count) - 1, len(spam_count))
	ax.bar(xline_spam, spam_count['Count'], width=0.75, color='b')
	ax.set_title('SMS Spam by length of message')
	plt.xlabel('length')
	plt.ylabel('frequency')
	plt.show()

	# Count the number of integer for Ham
	ham_integer = pd.DataFrame(pd.value_counts(ham['Number'], sort=True).sort_index())
	print("Ham count integer",ham_integer['Number'].count())

	# Count the number of integer for Spam
	spam_integer = pd.DataFrame(pd.value_counts(spam['Number'], sort=True).sort_index())
	print("Spam count integer",spam_integer['Number'].count())

	# Show SMS Ham by number of integer
	ax = plt.axes()
	xline_ham = np.linspace(0, len(ham_integer)-1, len(ham_integer))
	ax.bar(xline_ham, ham_integer['Number'], width=2.2, color='r')
	ax.set_title('SMS Ham by number integer')
	plt.xlabel('number')
	plt.ylabel('frequency')
	plt.show()

	#Show SMS Spam by number of integer
	ax = plt.axes()
	xline_spam = np.linspace(0, len(spam_integer)-1, len(spam_integer))
	ax.bar(xline_spam, spam_integer['Number'], width=2.2, color='b')
	ax.set_title('SMS Spam by number integer')
	plt.xlabel('number')
	plt.ylabel('frequency')
	plt.show()

	# Count the number of caracter for Ham
	ham_caractere = pd.DataFrame(pd.value_counts(ham['Caracter'], sort=True).sort_index())
	print("Ham count caracter",ham_caractere['Caracter'].count())

	# Count the number of caracter for spam
	spam_caractere = pd.DataFrame(pd.value_counts(spam['Caracter'], sort=True).sort_index())
	print("Spam count caracter",spam_caractere['Caracter'].count())

	# Show SMS Ham by number of caract
	ax = plt.axes()
	xline_ham = np.linspace(0, len(ham_caractere)-1, len(ham_caractere))
	ax.bar(xline_ham, ham_caractere['Caracter'], width=2.2, color='r')
	ax.set_title('SMS Ham by number special caractere')
	plt.xlabel('number')
	plt.ylabel('frequency')
	plt.show()

	# Show SMS Spam by number of caracter
	ax = plt.axes()
	xline_spam = np.linspace(0, len(spam_caractere)-1, len(spam_caractere))
	ax.bar(xline_spam, spam_caractere['Caracter'], width=2.2, color='b')
	ax.set_title('SMS Spam by number special caractere')
	plt.xlabel('number')
	plt.ylabel('frequency')
	plt.show()
