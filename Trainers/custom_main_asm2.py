import os
import copy
import click
import torch
import torch.utils.data
import bc_resnet_model
import get_data
import train
import apply
import util
# from QResNet182D import QResNet_182d
# from ResNet182D import ResNet_182d
# from QMatchboxnetsmall import QMatchBoxNet
# from QResNet181D import QResNet_181d
# from QMatchboxnet85k2dcomp import QMatchBoxNet85k2Dcompatible
# from QMatchboxnetnigbudget2Dcomp import QMatchBoxNetBB
# from bc_resnet_model_quaterionic import QBcResNetModel
from bc_resnet_model_quaterionicASM import QBcResNetModelASM

# from QMatchboxnetbigbudget import QMatchBoxNetBB

def run(model, train_loader, validation_loader, test_loader, optimizer, scheduler, device, checkpoint_file, n_epoch=10, log_interval=100):
    best_score = 0
    best_model = copy.deepcopy(model)
    for epoch in range(n_epoch):
        print(f"--- start epoch {epoch} ---")
        train.train_epoch_asm2(model, optimizer, train_loader, device, epoch, log_interval=log_interval)
        if scheduler:
            scheduler.step()
        score = apply.compute_accuracy_antisymemtric(model, validation_loader, device)
        print(f"Validation accuracy: {score:.5f}")
        if best_score < score:
            best_score = score
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), checkpoint_file)
    print(f"Top validation accuracy: {best_score:.5f}")
    test_score = apply.compute_accuracy_antisymemtric(best_model, test_loader, device)
    print(f"Test accuracy: {test_score:.5f}")


@click.group(help="Train and apply BC-ResNet Keyword Spotting Model")

def cli():
    pass


@cli.command("train", help="Train model")
@click.option("--scale", type=int, default=1, help="model width will be multiplied by scale")
@click.option("--batch-size", type=int, default=128, help="batch size")
@click.option("--device", type=str, default=util.get_device(), help="`cuda` or `cpu`")
@click.option("--epoch", type=int, default=10, help="number of epochs to train")
@click.option("--log-interval", type=int, default=100, help="display train loss after every `log-interval` batch")
@click.option("--checkpoint-file", type=str, default="model3.torch", help="file to save model checkpoint")
@click.option("--optimizer", type=str, default="adam", help="optimizer adam/sgd")
@click.option("--dropout", type=float, default=0.1, help="dropout")
@click.option("--subspectral-norm/--dropout-norm", type=bool, default=False, help="use SubspectralNorm or Dropout")
def train_command(scale, batch_size, device, epoch, log_interval, checkpoint_file, optimizer, dropout, subspectral_norm):
    if os.path.exists(checkpoint_file):
        raise FileExistsError(f"{checkpoint_file} already exists")

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    print(f"Device: {device}")
    print(f"Use subspectral norm: {subspectral_norm}")

    # model = bc_resnet_model.BcResNetModel(
    #     n_class=get_data.N_CLASS,
    #     scale=scale,
    #     dropout=dropout,
    #     use_subspectral=subspectral_norm,
    # ).to(device)
    
    # model = QResNet_182d(image_channels=4, num_classes=35).to(device)
    # model = ResNet_182d(image_channels=4,num_classes=35).to(device)
    
    # model = QResNet_181d(image_channels=160, num_classes=35).to(device)
    
    # model = QMatchBoxNet85k2Dcompatible(preprocessing_mode='off').to(device)
    
    # model = QMatchBoxNetBB(preprocessing_mode='off').to(device)
    
    # model = QBcResNetModel(n_class=35, scale=1,dropout=0.2,use_subspectral=False).to(device)
    
    model = QBcResNetModelASM(n_class=35, scale=6,dropout=0.2,use_subspectral=True).to(device)
    train_loader = torch.utils.data.DataLoader(
        get_data.SubsetSC(subset="training"),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=get_data.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    validation_loader = torch.utils.data.DataLoader(
        get_data.SubsetSC(subset="validation"),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=get_data.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        get_data.SubsetSC(subset="testing"),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=get_data.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
        scheduler = None
    else:
        raise ValueError(f"Unknown optimizer {optimizer}, use adam/sgd")

    run(
        model,
        train_loader,
        validation_loader,
        test_loader,
        optimizer,
        scheduler,
        device,
        checkpoint_file,
        n_epoch=epoch,
        log_interval=log_interval
    )


@cli.command("test", help="Test model accuracy on test set")
@click.option("--model-file", type=str, help="path to model weights")
@click.option("--scale", type=int, default=1, help="model width will be multiplied by scale")
@click.option("--batch-size", type=int, default=256, help="batch size")
@click.option("--device", type=str, default=util.get_device(), help="`cuda` or `cpu`")
@click.option("--dropout", type=float, default=0.1, help="dropout")
@click.option("--subspectral-norm/--dropout-norm", type=bool, default=False, help="use SubspectralNorm or Dropout")
def test_command(model_file, scale, batch_size, device, dropout, subspectral_norm):
    if not os.path.exists(model_file):
        raise FileExistsError(f"model {model_file} not exists")

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    print(f"Device: {device}")
    print(f"Use subspectral norm: {subspectral_norm}")

    # model = bc_resnet_model.BcResNetModel(
    #     n_class=get_data.N_CLASS,
    #     scale=scale,
    #     dropout=dropout,
    #     use_subspectral=subspectral_norm,
    # ).to(device)
    
    # model = QResNet_182d(image_channels=4, num_classes=35).to(device)
    
    # model = ResNet_182d(image_channels=4,num_classes=35).to(device)
    
    # model = QResNet_181d(image_channels=160, num_classes=35).to(device)
    
    # model = QMatchBoxNet85k2Dcompatible(preprocessing_mode='off').to(device)
    # model = QMatchBoxNetBB(preprocessing_mode='off')
    # model = QMatchBoxNetBB(preprocessing_mode='off').to(device)
    
    # model = QBcResNetModel(n_class=35, scale=1,dropout=0.2,use_subspectral=False)
    
    # model = QBcResNetModel(n_class=35, scale=1,dropout=0.2,use_subspectral=False).to(device)
    
    # model = QBcResNetModel(n_class=35, scale=8,dropout=0.2,use_subspectral=False).to(device)
    model = QBcResNetModelASM(n_class=35, scale=6,dropout=0.2,use_subspectral=True).to(device)
    
    model.load_state_dict(torch.load(model_file))

    test_loader = torch.utils.data.DataLoader(
        get_data.SubsetSC(subset="testing"),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=get_data.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_score = apply.compute_accuracy_antisymemtric(model, test_loader, device)
    print(f"Test accuracy: {test_score}")

##########ye niche wala code ek file pe lagane ke lie hain 

@cli.command("apply", help="Apply model to wav file")
@click.option("--model-file", type=str, help="path to model weights")
@click.option("--wav-file", type=str, help="path to wav sound file")
@click.option("--scale", type=int, default=1, help="model width will be multiplied by scale")
@click.option("--device", type=str, default=util.get_device(), help="`cuda` or `cpu`")
@click.option("--dropout", type=float, default=0.1, help="dropout")
@click.option("--subspectral-norm/--dropout-norm", type=bool, default=True, help="use SubspectralNorm or Dropout")
def apply_command(model_file, wav_file, scale, device, dropout, subspectral_norm):
    if not os.path.exists(model_file):
        raise FileExistsError(f"model file {model_file} not exists")
    if not os.path.exists(wav_file):
        raise FileExistsError(f"sound file {wav_file} not exists")

    model = bc_resnet_model.BcResNetModel(
        n_class=get_data.N_CLASS,
        scale=scale,
        dropout=dropout,
        use_subspectral=subspectral_norm,
    ).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    predictions = apply.apply_to_file(model, wav_file, device)
    for label, prob in predictions[:5]:
        print(f"{label}\t{prob:.5f}")


if __name__ == "__main__":
    cli()
