import urllib.request
import urllib.error
import time
from multiprocessing import Pool

start = time.time()

with open('links2.txt', 'r') as f:
    x = f.readlines()
    
urls = [y.rstrip().lower() for y in x]


def checkurl(url):
    try:
        conn = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        # Return code error (e.g. 404, 501, ...)
        # ...
        print('HTTPError: {}'.format(e.code) + ', ' + url)
        return None
    except urllib.error.URLError as e:
        # Not an HTTP-specific error (e.g. connection refused)
        # ...
        print('URLError: {}'.format(e.reason) + ', ' + url)
        return None
    else:
        # 200
        # ...
        print('good' + ', ' + url)
        return url


if __name__ == "__main__":
    p = Pool(processes=20)
    result = p.map(checkurl, x)

    with open('resultsLarge.txt','w') as out:
        for r in result:
            if r != None:
                out.write("%s" % r)

print("done in : ", time.time()-start)