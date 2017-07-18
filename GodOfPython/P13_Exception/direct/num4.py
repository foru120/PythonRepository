class UserException(Exception):
    def __str__(self):
        return 'There is p'

try:
    data = input('>')

    if 'p' in data:
        raise UserException
    print(data)
except UserException as e:
    print(e)