"""
1. Should spaces between words be left as emptry strings?
2. Need to go over all files to add all unique characters or just hardcode
   unique characters to the alphabet and specials ex. <eos>
3. But need to zero pad to the longest token length
4. Also we can either use indices or one hot encoding 
    --> using indices we can specify the number of classes in the embedding layer
"""

def parse_file(text):
    """
        - clean text file
        - split into list of chars
        - append <sos> and <eos>
       
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits))
    text = text.lower().strip()

    text = ['<space>' if char == ' ' else char for char in text]
    text.insert(0, '<sos>')
    text.append('<eos>')
    return text

def pad_file(text, max_length):
    """
        - zero pad to match max length
        - convert to index
        - return padded[:-1] --> decoder input and padded[1:] --> decoder output
    """

    padded = np.zeros(max_length)
    for i, char in enumerate(text):
        padded[i] = token_to_index[char]

    return padded[:-1], padded[1:]

def preprocess_text():
    """
        - open all text files
        - call parse file for each file
        - count the number of tokens to get max length
        - save results in respective folders (decoder input and decoder output)
    """

    file_names = os.listdir(path_to_text_files)
    cleaned_text = []
    
    for index, file in enumerate(file_names):
        with open(path_to_text_files + "/" + file) as f:
            for line in f:
                line = parse_file(line)
                cleaned_text.append(line)
        print("Files Cleaned: ", index + 1, "Files Remaining: ", len(file_names) - (index + 1))

    max_length = max([len(txt) for txt in cleaned_text])
    
    for index, text in enumerate(cleaned_text):
        decoder_input, decoder_output = pad_file(text, max_length)
        file = file_names[index].split(".")[0]
        np.save(path_to_decoder_input + "/" + "input_" +  file, decoder_input)
        np.save(path_to_decoder_ouput + "/" + "output_" + file, decoder_output)
        print("Files Done: ", index + 1, "Files Remaining: ", len(cleaned_text) - (index + 1))

preprocess_text()