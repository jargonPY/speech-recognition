from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

"""
  https://d35mpxyw7m7k7g.cloudfront.net/bigdata_1/Get+Authentication+for+Google+Service+API+.pdf
  https://stackoverflow.com/questions/24419188/automating-pydrive-verification-process
"""

gauth = GoogleAuth()           
drive = GoogleDrive(gauth)

def upload_file(files):
  for file in files:
    gfile = drive.CreateFile({'parents': [{'id': '1pzschX3uMbxU0lB5WZ6IlEEeAUE8MZ-t'}]})
    # Read file and set it as the content of this instance.
    gfile.SetContentFile(upload_file)
    gfile.Upload() # Upload the file.

def list_and_download_files():
  file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format('1cIMiqUDUNldxO6Nl-KVuS9SV-cWi9WLi')}).GetList()
  for file in file_list:
	  print('title: %s, id: %s' % (file['title'], file['id']))
  for i, file in enumerate(sorted(file_list, key = lambda x: x['title']), start=1):
    print('Downloading {} file from GDrive ({}/{})'.format(file['title'], i, len(file_list)))
    file.GetContentFile(file['title'])

def write_to_drive():
  # Create a GoogleDriveFile instance with title 'test.txt'.
  file1 = drive.CreateFile({'parents': [{'id': '1cIMiqUDUNldxO6Nl-KVuS9SV-cWi9WLi'}],'title': 'test.txt'})  
  # Set content of the file from the given string.
  file1.SetContentString('Hello World!') 
  file1.Upload()