def download_data():
  if not os.path.exists('./hotdog_nothotdog'):
    import gdown
    url = 'https://drive.google.com/uc?id=1hwyBl4Fa0IHihun29ahszf1M2cxn9TFk'
    gdown.download(url, './hotdog_nothotdog.zip', quiet=False)
    !unzip ./hotdog_nothotdog.zip > /dev/null
