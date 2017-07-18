from urllib.parse import urlparse

# Get domain name (tistory.com)
def get_domain_name(url):
    try:
        results = get_sub_domain_name(url).split('.')
        return results[-2] + '.' + results[-1]
    except:
        return ''

# Get Personal Blog Domain name (creativeworks.tistory.com)
def get_blog_domain_name(url):
    try:
        results = get_sub_domain_name(url).split('.')
        return results[-3] + '.' + results[-2] + '.' + results[-1]
    except:
        return ''

# Get sub domain name (ex - mail.google.com)
def get_sub_domain_name(url):
    try:
        return urlparse(url).netloc
    except:
        return ''