import requests
import zipfile
import shutil
import os


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                
if __name__ == '__main__':
    file_id = '1h6vVnlFpWk7G2dlzc9DZuKzUuqvloA25'
    file_path = os.path.join(os.getcwd(), 'ckpt.zip')
    print("download...")
    download_file_from_google_drive(file_id, file_path)
    
    with zipfile.ZipFile(file_path) as f:
        for file in f.namelist():
            f.extract(file, os.getcwd())
    
    shutil.move("ckpt/deeplab_model_best.pth.tar","deeplab_trimap/checkpoint/")
    shutil.move("ckpt/stylegan2-ffhq-config-f.pt","./") 
    shutil.move("ckpt/gca-dist-all-data.pth","gca_matting/checkpoints_finetune/") 
            
    