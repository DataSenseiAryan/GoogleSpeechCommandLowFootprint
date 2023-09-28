import torch.nn.functional as F
import torchaudio
import torch


def train_epoch(model, optimizer, train_loader, device, epoch, log_interval):
    model.train()

    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        data_first = torchaudio.functional.compute_deltas(data)
        data_second = torchaudio.functional.compute_deltas(data_first)
        data_third = torchaudio.functional.compute_deltas(data_second)
        # print(data.shape)
        # print(data_first.shape)
        # print(data_second.shape)
        # print(data_third.shape)
        quaternion_input = torch.cat([data,data_first, data_second, data_third], dim=1)
        # print(data.shape)
        # print(" input shape",quaternion_input.shape)
        # print("target sjape",target.shape)
        target = target.to(device)
        output = model(quaternion_input)
        loss = F.nll_loss(output.squeeze(), target)

        # print("output.shape",output.shape)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch}\tLoss: {loss.item():.4f}")

        losses.append(loss.item())

    return losses


def train_epoch_original(model, optimizer, train_loader, device, epoch, log_interval):
    model.train()

    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        
        # data_first = torchaudio.functional.compute_deltas(data)
        # data_second = torchaudio.functional.compute_deltas(data_first)
        # data_third = torchaudio.functional.compute_deltas(data_second)
        # print(data.shape)
        # print(data_first.shape)
        # print(data_second.shape)
        # print(data_third.shape)
        # quaternion_input = torch.cat([data,data_first, data_second, data_third], dim=1)
        # print(data.shape)
        # print(" input shape",quaternion_input.shape)
        # print("target sjape",target.shape)
        
        target = target.to(device)
        output = model(data)
        loss = F.nll_loss(output.squeeze(), target)
        
        # print("output.shape",output.shape)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch}\tLoss: {loss.item():.4f}")

        losses.append(loss.item())

    return losses

def train_epoch_asm(model, optimizer, train_loader, device, epoch, log_interval):
    model.train()

    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        zero_matrix = torch.zeros(data.size()).to(device)
        data_first = torchaudio.functional.compute_deltas(data)
        data_second = torchaudio.functional.compute_deltas(data_first)
        #data_third = torchaudio.functional.compute_deltas(data_second)
        # print(data.shape)
        # print(data_first.shape)
        # print(data_second.shape)
        # print(data_third.shape)
        quaternion_input = torch.cat([zero_matrix,data,data_first, data_second], dim=1)
        # print(data.shape)
        # print(" input shape",quaternion_input.shape)
        # print("target sjape",target.shape)
        target = target.to(device)
        output = model(quaternion_input)
        loss = F.nll_loss(output.squeeze(), target)

        # print("output.shape",output.shape)
        # orthoginal Asymmetric matrix initialisation ORGASM

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch}\tLoss: {loss.item():.4f}")

        losses.append(loss.item())

    return losses


def train_epoch_asm2(model, optimizer, train_loader, device, epoch, log_interval):
    model.train()

    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        #zero_matrix = torch.zeros(data.size()).to(device)
        data_first = torchaudio.functional.compute_deltas(data)
        data_second = torchaudio.functional.compute_deltas(data_first)
        data_third = torchaudio.functional.compute_deltas(data_second)
        # print(data.shape)
        # print(data_first.shape)
        # print(data_second.shape)
        # print(data_third.shape)
        quaternion_input = torch.cat([data,data_first, data_second, data_third], dim=1)
        # print(data.shape)
        # print(" input shape",quaternion_input.shape)
        # print("target sjape",target.shape)
        target = target.to(device)
        output = model(quaternion_input)
        loss = F.nll_loss(output.squeeze(), target)

        # print("output.shape",output.shape)
        # orthoginal Asymmetric matrix initialisation ORGASM

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch}\tLoss: {loss.item():.4f}")

        losses.append(loss.item())

    return losses


def train_epoch_phase(model, optimizer, train_loader, device, epoch, log_interval):
    model.train()

    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        
        
        
        data = data.to(device)
        
        
        # self.to_ifgram = 
        # freqs, times, mags = librosa.reassigned_spectrogram(waveform, sr=SAMPLE_RATE, S=None, n_fft=1024, hop_length=None, win_length=None, window='hann', center=True, reassign_frequencies=True, reassign_times=True, ref_power=1e-06, fill_nan=False, clip=True, dtype=None, pad_mode='constant')
        # log_mel = (self.to_mel(waveform) + EPS).log2()
        # freqs_delta = torchaudio.functional.compute_deltas(freqs)
        # freqs_delta = torchaudio.functional.compute_deltas(log_mel)
        # mags_db = librosa.amplitude_to_db(mags, ref=numpy.max)

        # data_first = torchaudio.functional.compute_deltas(data)
        # data_second = torchaudio.functional.compute_deltas(data_first)
        # data_third = torchaudio.functional.compute_deltas(data_second)
        # print(data.shape)
        # print(data_first.shape)
        # print(data_second.shape)
        # print(data_third.shape)
        # quaternion_input = torch.cat([data,data_first, data_second, data_third], dim=1)
        # print(data.shape)
        # print(" input shape",quaternion_input.shape)
        # print("target sjape",target.shape)
        target = target.to(device)
        output = model(data)
        loss = F.nll_loss(output.squeeze(), target)
        # print("output.shape",output.shape)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch}\tLoss: {loss.item():.4f}")
        losses.append(loss.item())
    return losses


def train(n_epoch, model, optimizer, train_loader, device, log_interval):
    print(f"--- Start train {n_epoch} epoches")
    for epoch in range(n_epoch):
        print(f"--- Start epoch {epoch+1}")
        train_epoch(model, optimizer, train_loader, device, epoch, log_interval)
    print("--- Done train")
    

def train_original(n_epoch, model, optimizer, train_loader, device, log_interval):
    print(f"--- Start train {n_epoch} epoches")
    for epoch in range(n_epoch):
        print(f"--- Start epoch {epoch+1}")
        train_epoch_original(model, optimizer, train_loader, device, epoch, log_interval)
    print("--- Done train")


def train_phase(n_epoch, model, optimizer, train_loader, device, log_interval):
    print(f"--- Start train {n_epoch} epoches")
    for epoch in range(n_epoch):
        print(f"--- Start epoch {epoch+1}")
        train_epoch_phase(model, optimizer, train_loader, device, epoch, log_interval)
    print("--- Done train ---")

def train_asm(n_epoch, model, optimizer, train_loader, device, log_interval):
    print(f"--- Start train {n_epoch} epoches")
    for epoch in range(n_epoch):
        print(f"--- Start epoch {epoch+1}")
        train_epoch_asm(model, optimizer, train_loader, device, epoch, log_interval)
    print("--- Done train")