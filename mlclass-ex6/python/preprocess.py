import tarfile
import re
import stemming
import csv
from stemming.porter2 import stem
from collections import Counter

N = 2000

def read_email_from_tar(tar_filename):

	tar = tarfile.open(tar_filename, "r:gz")

	for tarinfo in tar:

		if tarinfo.isfile() and re.match(r"^.*\/[^\.].+$", tarinfo.name):
			yield (tarinfo.name, tar.extractfile(tarinfo).read())



def preprocess_email(email):

	# strip headers
	hdrstart = email.find("\n\n")

	email = email[(hdrstart + 2):]


	# lower case
	email = email.lower()

	# strip html tags
	email = re.sub(r"<[^<>]+>", "", email)

	# handle numbers
	email = re.sub(r"[0-9]+", " number ", email)

	# handle urls
	email = re.sub(r"(http|https)://[^\s]*", " httpaddr ", email)

	# handle emails
	email = re.sub(r"[^\s]+@[^\s]+", " emailaddr ", email)

	# handle dollar sign
	email = re.sub(r"[$]+", " dollar ", email)


	# split words
	words = re.split(r"[ @\$\/#\.\-:&\*\+=\[\]?!\(\)\{\},'\">_<;%\n\r\t\|]+", email)


	# stem words
	words = map(stem, words)

	# remove duplicates
	words = set(words)

	# remove empty word
	words.discard("")

	return words


def make_features(words, vocab):

	x = [0] * N

	for (i, word) in enumerate(vocab):
		if word in email_words:
			x[i] = 1

	return x


def get_vocab():

	with open('vocab2.txt', 'rb') as csvfile:
		csvreader = csv.reader(csvfile, delimiter='	')
		vocab = [word for (word, count) in csvreader]

	return vocab



ham_emails = []
spam_emails = []

for (filename, email) in read_email_from_tar("ham.tar.gz"):

	# get words of email
	words = preprocess_email(email)
	ham_emails.append(words)

	print("%s %d" % (filename, len(words)))


for (filename, email) in read_email_from_tar("spam.tar.gz"):

	# get words of email
	words = preprocess_email(email)
	spam_emails.append(words)

	print("%s %d" % (filename, len(words)))


all_words = [item for sublist in (ham_emails + spam_emails) for item in sublist]
vocab = [word for (word, count) in Counter(all_words).most_common(N)]
print(vocab)

with open('vocab.txt', 'wb') as csvfile:

	csv_writer = csv.writer(csvfile, delimiter="	")

	for (i, word) in enumerate(vocab):
		csv_writer.writerow([i + 1, word])

with open('train.csv', 'wb') as csvfile:

	csv_writer = csv.writer(csvfile)

	for email_words in ham_emails:

		x = make_features(email_words, vocab)
		csv_writer.writerow(x + [0])

	for email_words in spam_emails:

		x = make_features(email_words, vocab)
		csv_writer.writerow(x + [1])



