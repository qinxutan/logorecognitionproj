import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import torchvision
import torch.nn.functional as F
from lossfunction.contrastive import ContrastiveLoss
from loaders.datasetBatch import SiameseNetworkDataset
from models import SNbetternet, SNlogonet, SNresnet18, SNdenseNet, SNinception, SNsqueeze, SNvgg, SNtests
from misc.misc import Utils
from params.config import Config
import csv
import os


class Trainer:

    def _get_class_name(path):
        # Extract the filename without extension
        filename = os.path.splitext(os.path.basename(path))[0]
        # Split the filename by hyphen and return the first part as class name
        return filename.split('-')[0]

    @staticmethod
    def train():
        print("Training process initialized...")
        print("dataset: ", Config.training_dir)

        nh_session, nh_seed = SiameseNetworkDataset.load_neural_hash_model()

        if nh_session is None or nh_seed is None:
            print("Failed to load neural hash model and seed. Exiting.")
            return

        siamese_dataset = SiameseNetworkDataset(
            root_dir=Config.training_dir,
            transform=transforms.Compose([transforms.Resize((Config.im_w, Config.im_h)), transforms.ToTensor()]),
            should_invert=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("CUDA is available. GPU support is enabled.")
            print("GPU device name:", torch.cuda.get_device_name(0))  # Prints the name of the GPU device
        else:
            print("CUDA is not available. GPU support is disabled.")

        print("lr:     ", Config.lrate)
        print("batch:  ", Config.train_batch_size)
        print("epochs: ", Config.train_number_epochs)

        net = Trainer.selectModel()  # Move the model to the appropriate device
        net = net.to(device)
        net.train()

        criterion = ContrastiveLoss()
        optimizer = optim.SGD(net.parameters(), lr=Config.lrate)

        counter = []
        loss_history = []

        best_loss = 10**15
        best_epoch = 0
        break_counter = 0  # break after 20 epochs without loss improvement
        matching_count = 0

        with open('comparison_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header row
            writer.writerow(['Image 1', 'Image 2', 'Class Name Ground Truth', 'Feature Ground Truth', 'OCR Score', 'ORB Feature', 'Neural Hash'])

            for epoch in range(0, Config.train_number_epochs):
                print("Epoch number: ", epoch)
                average_epoch_loss = 0
                num_batches = len(siamese_dataset) // Config.train_batch_size

                for i in range(num_batches):
                    batch_start_index = i * Config.train_batch_size
                    batch_end_index = batch_start_index + Config.train_batch_size

                    if batch_end_index > len(siamese_dataset):
                        continue

                    batch_data = [siamese_dataset[j] for j in range(batch_start_index, batch_end_index)]

                    img0, img1, feature_vector = zip(*batch_data)
                    feature_vector = torch.stack([torch.from_numpy(fv).float() for fv in feature_vector])
                    img0 = torch.stack(img0).to(device)
                    img1 = torch.stack(img1).to(device)
                    feature_vector = feature_vector.float().to(device)

                    img0 = img0.view(-1, 3, Config.im_h, Config.im_w)
                    img1 = img1.view(-1, 3, Config.im_h, Config.im_w)

                    scores = net(img0, img1, feature_vector)
                

                    labels = [1 if fv[-1] > 0.375 else 0 for fv in feature_vector.cpu().numpy()]
                    labels = torch.tensor(labels).float().to(device)

                    optimizer.zero_grad()

                    if Config.bceLoss:
                        loss = F.binary_cross_entropy_with_logits(scores, labels)
                    else:  # contrastive
                        loss = criterion(scores, labels)

                    if torch.isnan(loss).any():
                            print(f"NaN detected in loss at epoch {epoch}, batch {i}")
                            continue
                
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()

                    average_epoch_loss += loss.item()

                    img_index_0 = i * Config.train_batch_size
                    img_index_1 = img_index_0 + 1

                    if img_index_1 < len(siamese_dataset.image_paths):
                        print(f"Pair comparison: {siamese_dataset.image_paths[img_index_0]} vs. {siamese_dataset.image_paths[img_index_1]}")
                    else:
                        print(f"Pair comparison: {siamese_dataset.image_paths[img_index_0]} vs. None (out of range)")

                    print("Header: ocr_score, orb_feature, neural_hash")
                    print(f"Row: {feature_vector[0]}, {feature_vector[1]}, {feature_vector[2]}")

                    feature_lists = [fv.tolist() for fv in feature_vector]

                average_epoch_loss = average_epoch_loss / len(siamese_dataset)
                print("Epoch number {}\n Current loss {}\n".format(epoch, average_epoch_loss))
                counter.append(epoch)
                loss_history.append(average_epoch_loss)

                if epoch % 10 == 0:
                    torch.save(net.state_dict(), Config.best_model_path + str(epoch)+ '.pth')

                if average_epoch_loss < best_loss:
                    best_loss = average_epoch_loss
                    best_epoch = epoch
                    torch.save(net.state_dict(), Config.best_model_path)
                    print("------------------------Best epoch: ", epoch)
                    break_counter = 0

                if break_counter >= 20:
                    print("Training break...")
                    break

                break_counter += 1

        torch.save(net.state_dict(), Config.model_path)
        Utils.show_plot(counter, loss_history)

    @staticmethod
    def selectModel():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if Config.model == "resnet":
            return SNresnet18.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).to(device)
        elif Config.model == "logonet":
            return SNlogonet.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).to(device)
        elif Config.model == "dense":
            return SNdenseNet.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).to(device)
        elif Config.model == "inception":
            return SNinception.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).to(device)
        elif Config.model == "vgg":
            return SNvgg.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).to(device)
        elif Config.model == "squeeze":
            return SNsqueeze.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).to(device)
        elif Config.model == "betternet":
            return SNbetternet.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).to(device)
        elif Config.model == "tests":
            return SNtests.SiameseNetwork(pretrained=Config.pretrained).to(device)