def train_one_epoch(model, optimizer, loader, epoch, mode="train"):
        epoch_loss = 0
        model.train()
        
        for batch in loader:
            imgs = []
            targets = []

            for sample in batch:
                imgs.append(sample[0].to(device))
                target = {}
                target["boxes"] = sample[1]["boxes"].to(device)
                target["labels"] = sample[1]["labels"].to(torch.int64).to(device)
                targets.append(target)

            ret_dict = model(imgs, targets)
            loss = sum(v for v in ret_dict.values())
            epoch_loss += loss.cpu().detach().numpy()

            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.update(1)

            writer.add_scalar("Loss/" + mode, epoch_loss, epoch)

        return epoch_loss