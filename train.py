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
import yaml

def save_config(config, save_path):
    """Save the configuration to a YAML file"""
    config_dict = {}
    for section in ['data', 'model', 'training', 'preprocess', 'loss']:
        section_config = getattr(config, section)
        section_dict = vars(section_config)
        # Filter out any special attributes
        section_dict = {k: v for k, v in section_dict.items() if not k.startswith('_')}
        config_dict[section] = section_dict

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--exp_name', type=str, default='experiment', help='Experiment name')
    parser.add_argument('--resume', type=str, help='Path to checkpoint for resuming')
    parser.add_argument('--monitor_samples', type=int, default=50,
                        help='Number of samples to monitor per epoch')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for dataset (overrides config)')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size (overrides config)')
    parser.add_argument('--num_workers', type=int, required=True, help='Number of data loader workers (overrides config)')
    parser.add_argument('--prefetch_factor', type=int, required=True, help='Prefetch factor for data loader (overrides config)')

    # 添加监控器启用/禁用的参数
    parser.add_argument('--enable_monitor', action='store_true', help='Enable data flow monitoring')
    parser.add_argument('--disable_monitor', action='store_false', dest='enable_monitor',
                        help='Disable data flow monitoring')
    parser.set_defaults(enable_monitor=False)  # 默认关闭监控器

    return parser.parse_args()

def main():

    torch.set_float32_matmul_precision('medium')

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = Config.load(args.config)

    # Override config values if specified in command-line arguments
    if args.root_dir:
        config.data.root_dir = args.root_dir
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.num_workers:
        config.data.num_workers = args.num_workers
    if args.prefetch_factor:
        config.data.prefetch_factor = args.prefetch_factor


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

    # Save the configuration with any overrides applied
    config_save_path = run_dir / 'config.yaml'
    save_config(config, config_save_path)

    # 初始化监控器
    print("Initializing data flow monitor...")
    if args.enable_monitor:
        monitor = DataFlowMonitor(
            root_dir=str(monitor_dir),
            max_samples=args.monitor_samples
        )
        print("Data flow monitor enabled.")
    else:
        monitor = None
        print("Data flow monitor disabled.")

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
            save_top_k=1,
            monitor='val_loss',
            mode='min',
            save_last=False
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
        if monitor is not None:
            monitor.close()
            print("Monitor closed successfully")
        else:
            print("No monitor to close")

if __name__ == "__main__":
    main()