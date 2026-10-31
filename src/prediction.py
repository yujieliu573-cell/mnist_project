import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split
new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [
    (f"{new_mirror}/{url.split('/')[-1]}", md5)
    for url, md5 in datasets.MNIST.resources
]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081,))
])
print("正在下载/加载数据集...")
full_train = datasets.MNIST('./data',train=True,download=True,transform=transform)
test_dataset = datasets.MNIST('./data',train=False,download=True,transform=transform)

train_size = int(0.8*len(full_train))
val_size =len(full_train) - train_size
train_dataset,val_dataset=random_split(full_train,[train_size,val_size])

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=1000,shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

data,target = next(iter(train_loader))
print(f"数据加载成功！输入形状：{data.shape},标签形状：{target.shape}")

import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
    def forward(self,x):
        x=self.flatten(x)
        return self.layers(x)
model = MLP()
output = model(data)
print(f"模型搭建成功输入形状：{data.shape},输出形状：{output.shape}")


import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
train_losses = []
val_accuracies=[]
step_count=0
epochs = 5
print("开始训练")
for epoch in range(epochs):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader) :
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        step_count += 1
        train_losses.append(loss.item())
        if step_count % 100 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for val_data,val_target in val_loader:
                    val_output = model(val_data)
                    pred = val_output.argmax(dim=1)
                    correct +=(pred == val_target).sum().item()
                acc = correct/len(val_dataset)
                val_accuracies.append(acc)
                print(f"Epoch {epoch+1} | Step {step_count} | Loss: {loss.item():.4f} | Val Acc: {acc:.4f}")
                model.train()

import matplotlib.pyplot as plt
print("训练结束，正在绘制曲线...")
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses,label='Train',color='blue')
plt.title('Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1,2,2)
val_steps = [i*100 for i in range(1,len(val_accuracies)+1)]
plt.plot(val_steps,val_accuracies,label='Val Accuracy',color='green')
plt.title('Validation Accuracy')
plt.xlabel('Steps')
plt.legend()

plt.tight_layout()
plt.savefig('baseline_curves.png')
print("训练已完成!曲线已保存为baseline_curves.png")
# ================= 6. 实战：让模型认字并展示图片 =================
print("\n--- 让我们来看看模型到底认出了什么！ ---")
images, labels = next(iter(test_loader))
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# 把 10 张图片画出来，标题写上预测结果
plt.figure(figsize=(15, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    # 把 tensor 转回图片格式
    img = images[i].squeeze().numpy()
    plt.imshow(img, cmap='gray')
    
    true_num = labels[i].item()
    pred_num = predicted[i].item()
    color = 'green' if true_num == pred_num else 'red'
    plt.title(f"预测:{pred_num}\n真实:{true_num}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.savefig('predictions.png')
print("✅ 预测结果图片已保存为 predictions.png")
