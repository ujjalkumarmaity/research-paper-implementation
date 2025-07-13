from model import *


def validate(model: CBOWModel, val_dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for idx, (context, target) in enumerate(val_dataloader):
            context = context.to(model.project_layer.weight.device)
            target = target.to(model.project_layer.weight.device)
            output = model(context)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(val_dataloader)


def train(model: CBOWModel, train_dataloader, val_dataloader, config: Config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    train_loss = []
    val_loss = []

    for epoch in range(config.epoch):
        total_loss = 0
        model.train()
        for idx, (context, target) in enumerate(train_dataloader):
            context = context.to(model.project_layer.weight.device)
            target = target.to(model.project_layer.weight.device)

            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # if idx % 100 == 0:
            #     print(f'Epoch {epoch+1}, Batch {idx}, Loss: {loss.item()}')
        vloss = validate(model, val_dataloader)
        train_loss.append(total_loss / len(train_dataloader))
        val_loss.append(vloss)
        print(
            f"Epoch: {epoch+1}, Train Loss: {total_loss/len(train_dataloader)}, Valdation loss: {vloss}"
        )

    return {"train_loss": train_loss, "val_loss": val_loss}
