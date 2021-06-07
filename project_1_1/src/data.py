#def download_data():
#  if not os.path.exists('./hotdog_nothotdog'):
#    import gdown
#    url = 'https://drive.google.com/uc?id=1hwyBl4Fa0IHihun29ahszf1M2cxn9TFk'
#    gdown.download(url, './hotdog_nothotdog.zip', quiet=False)
#    !unzip ./hotdog_nothotdog.zip > /dev/null
    
class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='hotdog_nothotdog'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y
    
def get_data():
  size = 128
  train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                      transforms.ToTensor()])
  test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                      transforms.ToTensor()])

  batch_size = 64
  trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
  train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
  testset = Hotdog_NotHotdog(train=False, transform=test_transform)
  test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)
  
def plot_data():
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(20,10))
    
    for i in range(21):
        plt.subplot(5,7,i+1)
        plt.imshow(np.swapaxes(np.swapaxes(images[i].numpy(), 0, 2), 0, 1))
        plt.title(['hotdog', 'not hotdog'][labels[i].item()])
        plt.axis('off')
