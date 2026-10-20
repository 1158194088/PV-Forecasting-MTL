import os
import pytorch_lightning as pl
import math
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, Subset

# =========================================================================
from src.dataset.dataset_QuadTask import PVQuadTaskDataset
from src.model.QuadTask_final_pro import PVTransformer_QuadTask_Advanced
from src.evaluator.Pic_evaluator import evaluate_model_comprehensive
# =========================================================================

class HardcoreCheckpoint(pl.Callback):
    def __init__(self, dirpath):
        super().__init__()
        self.dirpath = dirpath
        self.best_mae = float('inf')
        self.best_model_path = ""

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        
        if 'val_mae' in metrics:
            current_mae = metrics['val_mae'].item()

            if math.isnan(current_mae) or math.isinf(current_mae):
                print(f"\n[WARNING] Abnormal val_mae ({current_mae}) detected. Emergency skip of current save to preserve existing model!")
                return
            last_path = os.path.join(self.dirpath, "last.ckpt")
            trainer.save_checkpoint(last_path)

            # =====================================================================================
            if current_mae < self.best_mae:
                self.best_mae = current_mae
                self.best_model_path = os.path.join(self.dirpath, "test_best_model.ckpt")
                trainer.save_checkpoint(self.best_model_path)
                print(f"\n[FORCE SAVE] New best val_mae: {current_mae:.4f}! Weights safely written to disk.")

def seed_everything(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)

if __name__ == "__main__":
    # ======================================================================================
    seed_everything(0)
    # ======================================================================================

    BASE_DIR = "PV_QuadTask_Project/data"
    MTL_FILE = os.path.join(BASE_DIR, "MTL_Dataset_pro.csv")
    NWP_FILE = os.path.join(BASE_DIR, "nwp_filtered_clean_1H_new.csv")
    DAILY_LABEL_FILE = os.path.join(BASE_DIR, "历史天级别-处理.csv")

    train_ratio = 0.7
    
    dataset = PVQuadTaskDataset(MTL_FILE, NWP_FILE, DAILY_LABEL_FILE, train_ratio=train_ratio)
    
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(0.1 * total_len)
    
    train_loader = DataLoader(Subset(dataset, range(0, train_len)), batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(dataset, range(train_len, train_len + val_len)), batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(Subset(dataset, range(train_len + val_len, total_len)), batch_size=32, shuffle=False, num_workers=4)

    # ======================================================================================
    model = PVTransformer_QuadTask_Advanced(
        nwp_dim=dataset.nwp_dim, 
        num_classes=dataset.num_classes, 
        d_model=128, 
        pred_len=24
    )
    # ======================================================================================
    
    early_stop = EarlyStopping(monitor="val_mae", patience=20, mode="min")
    
    CKPT_DIR = 'PV_QuadTask_Project/checkpoints/'
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    my_checkpoint = HardcoreCheckpoint(dirpath=CKPT_DIR)

    trainer = pl.Trainer(
        max_epochs=1000, 
        callbacks=[early_stop, my_checkpoint],
        accelerator="auto", 
        devices=1, 
        log_every_n_steps=5,
        num_sanity_val_steps=0
    )
    
    # ==============================================================================================
    print("Begin:test")
    trainer.fit(model, train_loader, val_loader)
    # ==============================================================================================

    best_model_path = my_checkpoint.best_model_path            
    # ==============================================================================================
    if best_model_path and os.path.exists(best_model_path): 
        print(f"\nLoading best validation weights for testing: {best_model_path}")
        best_model = PVTransformer_QuadTask_Advanced.load_from_checkpoint(
            checkpoint_path=best_model_path,
            nwp_dim=dataset.nwp_dim, 
            num_classes=dataset.num_classes, 
            d_model=128, 
            pred_len=24
        )
    else:
        print("\nWaring：Can't find flies!")
        best_model = model
    # ==============================================================================================

    # ======================================================================================
    evaluate_model_comprehensive(best_model, test_loader, dataset, 'test')
    # ======================================================================================