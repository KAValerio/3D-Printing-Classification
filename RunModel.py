from os import listdir
import pickle
from MakePredFunc import makePred

predvid = 6  # Which video to make a prediction on (numbers 1 through 6)
ppc = 12  # HOG parameter (pixels per cell)
cpb = 3  # HOG parameter (cells per block)
n = 20 ** 2  # Downsampled size of image


def numericalSort(value):
    import re
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

d = {1: 22, 2: 30, 3: 25, 4: 37, 5: 32, 6: 15}
dire = 'data/image_stream/video' + str(predvid) + '/'
mod = pickle.load(open("SGDModel.pkl", 'rb'))
otpred = []
otprob = []
otpredi = []
for i, file in enumerate(sorted(listdir(dire), key=numericalSort)):
    path = dire + file
    if i in [1, 15, 25, 38]:
        a, b = makePred(path, model=mod, ppc=ppc, cpb=cpb, n=n, figures=True)
    else:
        a, b = makePred(path, model=mod, ppc=ppc, cpb=cpb, n=n, figures=False)
    otpred.append(a)
    otprob.append(b)
    otpredi.append((i, a))
ots = [k for k in range(len(otpredi)) if otpredi[k][1] == 1]

inca = 0
incb = len(otpredi) - d[predvid]

for k in ots:
    if k < d[predvid]:
        inca += 1
    if k > d[predvid]:
        incb -= 1
print(round(inca / d[predvid] * 100), "% misclassified below change (", inca, "out of", str(d[predvid]), "values)")
print(round(incb / (len(otpredi) - d[predvid]) * 100), "% misclassified above change (", incb, "out of", \
      (len(otpredi) - d[predvid]), "values)")
if d[predvid] in ots:
    print("Error point identified correctly")
elif len(ots) == 0:
    print("No failure detected")
else:
    print("Identified change at", sorted(k for k in ots if k > d[predvid])[0], "when the change occured at", d[predvid])
