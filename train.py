import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from datetime import datetime
from pathlib import Path
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
import sys

from config.config import Config
from models.avse_model import AVSEModel
from data.data_module import AVSEDataModule
from utils.monitor import DataFlowMonitor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--resume', type=str, help='Path to checkpoint for resuming')
    parser.add_argument('--monitor_samples', type=int, default=50,
                        help='Number of samples to monitor per epoch')
    return parser.parse_args()

def main():

    torch.set_float32_matmul_precision('medium')

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = Config.load(args.config)

    # Set random seed
    pl.seed_everything(42)

    # Create directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(config.training.checkpoint_dir) / f'run_{timestamp}'
    checkpoint_dir = run_dir / 'checkpoints'
    log_dir = run_dir / 'logs'
    monitor_dir = run_dir / 'monitor_logs'

    for dir_path in [checkpoint_dir, log_dir, monitor_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Initialize monitor
    print("Initializing data flow monitor...")
    monitor = DataFlowMonitor(
        root_dir=str(monitor_dir),
        max_samples=args.monitor_samples
    )

    # Initialize data module with monitor
    print("Initializing data module...")
    data_module = AVSEDataModule(config, monitor=monitor)

    # Initialize model with monitor
    print("Initializing model...")
    model = AVSEModel(config, monitor=monitor)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='avse-{epoch:02d}-{val_loss_total:.4f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name='tensorboard',
        version='.',
        default_hp_metric=False
    )

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=config.training.num_epochs,
        precision=config.training.precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip,
        log_every_n_steps=50,
        enable_progress_bar=True,
        num_sanity_val_steps=2,  # 减少验证步骤
        enable_model_summary=True,  # 显示模型摘要
    )

    try:
        # Start training
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
    finally:
        # 确保监控器正确关闭
        monitor.close()
        print("Monitor closed successfully")

if __name__ == "__main__":
    main()