import fileinput


line_n = 0

learning_rate = ''
momentum = ''
activation = ''

train_averages = []
eval_averages = []
train_correct = []
eval_correct = []

for line in fileinput.input():

    tokens = line.split()
    tokens = [e for token in tokens for e in token.split('=')]
    if len(tokens) != 0:
        if line_n == 0:  # Creating perceptron... line
            learning_rate = tokens[4][:-1]
            momentum = tokens[6][:-1]
            activation = tokens[9]

        elif line_n % 3 == 2:  # Average
            eval_averages.append(tokens[2])
            eval_correct.append(tokens[5])

        elif line_n % 3 == 0:  # Train average loss
            train_averages.append(tokens[3])
            train_correct.append(tokens[6])

    line_n += 1

print('Learning rate:\t' + learning_rate)
print('Momentum:\t' + momentum)
print('Activation:\t' + activation)

print('Epoch\tAverage-eval\tCorrect-eval\tAverage-train\tCorrect-train')

i = 0
while i < len(eval_averages) or i < len(train_averages):
    print(str(i) + '\t' +
          (eval_averages[i] if i < len(eval_averages) else '') + '\t' +
          (eval_correct[i] if i < len(eval_correct) else '') + '\t' +
          (train_averages[i] if i < len(train_averages) else '') + '\t' +
          (train_correct[i] if i < len(train_correct) else ''))
    i += 1
