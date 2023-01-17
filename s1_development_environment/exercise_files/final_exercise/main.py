import argparse
import sys

import torch
from torch import nn, optim
import click

from data import mnist
from model import MyAwesomeModel


#@click.group()
# def cli():
#     pass


# @click.command()
# @click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    trainloader, _ = mnist()
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    epochs = 30
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1).float()
            

            # TODO: Training pass
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            print(f"Training loss: {running_loss/len(trainloader)}")
            torch.save(model.state_dict(), r'/home/student/Desktop/MLOPS_Final/S1_Development/M4_DLsoftware/s1_development_environment/exercise_files/final_exercise/checkpoint.pth')


# @click.command()
# @click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    with torch.no_grad():
        state_dict = torch.load(model_checkpoint)
        model = MyAwesomeModel()
        model.load_state_dict(state_dict)

        _, testloader = mnist()
        total = 0
        correct = 0

        for images, labels in testloader:
            output = model(images.float())
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        else:
            ## TODO: Implement the validation pass and print out the validation accuracy
            accuracy = correct/total
            print(f'Accuracy: {accuracy*100}%')

# cli.add_command(train)
# cli.add_command(evaluate)

if __name__ == "__main__":
    # cli()
    # train(1e-3)    
    evaluate(r'/home/student/Desktop/MLOPS_Final/S1_Development/M4_DLsoftware/s1_development_environment/exercise_files/final_exercise/checkpoint.pth')