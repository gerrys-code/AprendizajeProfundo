import os

name_files = os.listdir(".")
name_directory = (os.path.relpath(".",".."))


for file_name in name_files:
    sp = (file_name.split("_"))
    new_name = sp[0]+"."+name_directory+"."+sp[1]
    os.rename(file_name,new_name)

print ("Rename all files")
