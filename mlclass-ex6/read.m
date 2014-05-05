ham_files = readdir("spamassassin/ham");

n = 1899;
X = [];

for i=1:numel(ham_files)

	ham_filename = ham_files{i};

	if ham_filename(1) != "."

		ham_filename = ["spamassassin/ham/" ham_filename];

		file_contents = readFile(ham_filename);
		word_indices = processEmail(file_contents);

		x = zeros(n, 1);

		for i=word_indices
			x(i) = 1;
		end

		X = [X ; x];

		fprintf("%s\n", ham_filename);

	end

end

size(X)