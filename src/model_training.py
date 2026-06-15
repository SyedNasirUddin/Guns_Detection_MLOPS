import os
import torch
from torch.utils.data import DataLoader,random_split
from torch import optim
from src.model_architecture import FasterRCNNModel
from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_processing import GunDataset
from torch.utils.tensorboard import SummaryWriter
import time

logger = get_logger(__name__)

model_save_path = "artifacts/models/"
os.makedirs(model_save_path,exist_ok=True)

class ModelTraining:
    def __init__(self, model_class , num_classes , learning_rate ,epochs , dataset_path , device):
        self.model_class = model_class
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.device = device

        #### Tensorboard
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"tensorboard_logs/{timestamp}"
        os.makedirs(self.log_dir,exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        try:
            self.model = self.model_class(self.num_classes,self.device).model
            self.model.to(self.device)
            logger.info("Model Moved to device")

            self.optimizer = optim.Adam(self.model.parameters() , lr=self.learning_rate)
            logger.info("Optimizer has been initialized..")
        except Exception as e:
            logger.error(f"Failed to initilize model training {e}")
            raise CustomException("Failed to initialize model training " , e)
    
    def collate_fn(self,batch):
        return tuple(zip(*batch))
    
    def split_dataset(self):
        try:
            dataset = GunDataset(self.dataset_path)

            dataset = torch.utils.data.Subset(dataset,range(100))

            train_size = int(0.8*len(dataset))
            val_size = len(dataset) - train_size

            train_dataset , val_dataset = random_split(dataset , [train_size,val_size])

            train_loader = DataLoader(train_dataset , batch_size=1 , shuffle=True , num_workers=0 , collate_fn = self.collate_fn)
            val_loader = DataLoader(val_dataset , batch_size=1 , shuffle=False , num_workers=0 , collate_fn = self.collate_fn)

            logger.info("Data splitted sucesfuly...")
            return train_loader,val_loader
        
        except Exception as e:
            logger.error(f"Failed to do splitting data {e}")
            raise CustomException("Failed to do splitting data " , e)
    def train(self):
        try:

            print("ENTERED TRAIN METHOD")
            print("CUDA AVAILABLE =", torch.cuda.is_available())
            print("DEVICE =", self.device)

            train_loader, val_loader = self.split_dataset()

            print("DATA LOADER CREATED")
            print("TRAIN BATCHES =", len(train_loader))
            print("VAL BATCHES =", len(val_loader))

            for epoch in range(self.epochs):

                logger.info(f"Starting epoch {epoch}")

                print(f"\n{'='*50}")
                print(f"EPOCH {epoch}")
                print(f"{'='*50}\n")

                self.model.train()

                for i, (images, targets) in enumerate(train_loader):

                    images = [
                                img.to(self.device)
                                for img in images
                            ]

                    targets = [
                                {
                                    k: v.to(self.device)
                                    for k, v in t.items()
                                }
                                for t in targets
                            ]


                    print(f"\nBATCH {i+1}/{len(train_loader)}")

                    print("IMAGE TYPE =", type(images))
                    print("NUM IMAGES =", len(images))

                    for idx, img in enumerate(images):
                        print(
                            f"IMAGE {idx} SHAPE = {img.shape}"
                        )

                    print("TARGETS =", targets)

                    self.optimizer.zero_grad()

                    print("BEFORE MODEL")

                    with torch.autograd.set_detect_anomaly(True):

                        losses = self.model(
                            images,
                            targets
                        )

                    print("AFTER MODEL")

                    if isinstance(losses, dict):

                        total_loss = sum(
                            loss for loss in losses.values()
                        )

                        print(f"LOSS = {total_loss.item():.4f}")

                        for key, value in losses.items():
                            print(f"{key} = {value.item():.4f}")

                        self.writer.add_scalar(
                            "Loss/train",
                            total_loss.item(),
                            epoch * len(train_loader) + i
                        )

                    else:

                        total_loss = losses[0]

                    print(f"LOSS = {total_loss.item():.4f}")


                    print("BEFORE BACKWARD")
                    total_loss.backward()

                    print("AFTER BACKWARD")

                    print("OPTIMIZER STEP START")



                    self.optimizer.step()

                    print("OPTIMIZER STEP END")

                    
                print("AFTER OPTIMIZER STEP")

                print("TRAINING LOOP COMPLETE")

                self.writer.flush()

                print("STARTING VALIDATION")

                self.model.eval()

                with torch.no_grad():

                    for images, targets in val_loader:
                            
                        images = [
                                img.to(self.device)
                                for img in images
                            ]

                        targets = [
                                {
                                    k: v.to(self.device)
                                    for k, v in t.items()
                                }
                                for t in targets
                            ]


                        print("VALIDATION BATCH")
                        try: 

                            val_losses = self.model(
                                images,
                                targets
                            )

                            print(
                                "VAL LOSSES =",
                                val_losses
                            )
                        except Exception as e:
                            print(
                                    "Validation loss not available:",
                                    e
                                )

                model_path = os.path.join(
                                model_save_path,
                                "fasterrcnn.pth"
                            )

                torch.save(
                                self.model.state_dict(),
                                model_path
                            )

                print(
                                "MODEL SAVED TO:",
                                model_path
                            )

                logger.info(
                                "Model saved successfully"
                            )

        except Exception as e:

            print("EXCEPTION OCCURRED")
            print(type(e))
            print(e)

            logger.error(
                f"Failed to train model {e}"
            )

            raise CustomException(
                "Failed to train model",
                e
            )
    

if __name__=="__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("CUDA =", torch.cuda.is_available())
    print("DEVICE =", device)

    training = ModelTraining(
        model_class=FasterRCNNModel,
        num_classes=2,
        learning_rate=0.0001,
        dataset_path="artifacts/raw/",
        device=device,
        epochs=1
    )

    print("STARTING TRAINING")

    training.train()

    print("TRAINING FINISHED")
    