import subprocess

def is_number(s, number_type):
    """ Returns True is string is a number. """
    try:
        number_type(s)
        return True
    except ValueError:
        return False


# from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()


# from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()