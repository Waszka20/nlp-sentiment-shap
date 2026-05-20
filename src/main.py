import torch
from torch.utils.data import DataLoader, random_split
from dataset import TrumpTweetDataset


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    dataset = TrumpTweetDataset(threshold=0.25,overwrite=False,device=device)

    # 2 - positive, 1 - neutral, 0 - negative
    #print(dataset.df["label"].value_counts())

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,)
    


    embeddings, labels = next(iter(train_loader))
    print(embeddings.shape)
    print(labels.shape)


