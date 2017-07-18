import os

# Each website you want to crawl is a separate project(folder)
def create_project_dir(directory):
    if not os.path.exists(directory):
        print('Creating Project   ' + directory)
        os.makedirs(directory)

# Create queue and crawled files(if not created)
def create_data_files(project_name, base_url):
    queue = project_name + '/queue.txt'
    crawled = project_name + '/crawled.txt'

    if not os.path.isfile(queue):
        write_file(queue, base_url)
    if not os.path.isfile(crawled):
        write_file(crawled, '')

# Create a new file
def write_file(path, data):
    f = open(path, 'w')
    f.write(data)
    f.close()

# Add data onto an existing file
def append_to_file(path, data):
    with open(path, 'a') as file:
        file.write(data+'\n')

# Delete the contents of a file
def delete_file_contents(path):
    with open(path, 'w'):
        # for doing nothing
        pass

# Read a file and convert each line to set items
def file_to_set(file_name):
    results = set()
    with open(file_name, 'rt') as file:
        for line in file:
            results.add(line.replace('\n', '')) # 개행문자(\n)를 ''로 변환
    return results

# Iterate through a set, each item will be a new line in the file
def set_to_file(links, file):
    delete_file_contents(file)
    for link in sorted(links):
        append_to_file(file, link)