from queue import Queue

from general import *
from spider import Spider

from test.domain import *

PROJECT_NAME = 'creativeworks'
HOMEPAGE = 'https://creativeworks.tistory.com/'

# If you have your own web site, use get_domain_name(HOMEPAGE)

DOMAIN_NAME = get_blog_domain_name(HOMEPAGE)
QUEUE_FILE = PROJECT_NAME + '/queue.txt'
CRAWLED_FILE = PROJECT_NAME + '/crawled.txt'
NUMBER_OF_THREADS = 4
queue = Queue()
Spider(PROJECT_NAME, HOMEPAGE, DOMAIN_NAME)