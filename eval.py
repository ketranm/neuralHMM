import os,sys,math

# Format:  one sentence per line.  Space separated POS tags
if len(sys.argv) < 3:
  print "python eval.py predicted.txt gold.txt"
  print "Format:  One sentence / line.  Space separated POS tags"
  sys.exit()

G = []
for line in open(sys.argv[2],'r'):
  G.append(line.strip().split())

P = []
for line in open(sys.argv[1],'r'):
  P.append(line.strip().split())

## Assertions and tagsets
Gold = set()
Pred = set()
Total = 0.0
assert len(G) == len(P), "Lengths don't match %d %d" % (len(G), len(P))
for i in range(len(G)):
  assert len(G[i]) == len(P[i]), "Sentence %d lengths don't match" % i
  Gold.update(G[i])
  Pred.update(P[i])
  Total += len(G[i])

## Create Confusion Matrix
C = {}
for gold in Gold:
  C[gold] = {}
  for pred in Pred:
    C[gold][pred] = 0.0
for i in range(len(G)):
  for j in range(len(G[i])):
    C[G[i][j]][P[i][j]] += 1


## 1 to 1 evaluation
Mapping = []
Pairs = []
for gold in C:
  for pred in C[gold]:
    Pairs.append((C[gold][pred], gold, pred))
Pairs.sort()
Pairs.reverse()
Used_Gold = set()
Used_Pred = set()
Correct = 0
for count, gold, pred in Pairs:
  if gold not in Used_Gold and pred not in Used_Pred:
    Used_Gold.add(gold)
    Used_Pred.add(pred)
    Correct += count
    Mapping.append((gold,pred))
print "1-1: %6.3f" % (100.0*Correct/Total)

## Many to 1 evaluation
Used_Pred = set()
Correct = 0
for count, gold, pred in Pairs:
  if pred not in Used_Pred:
    Used_Pred.add(pred)
    Correct += count
print "M-1: %6.3f" % (100.0*Correct/Total)

## VM evaluation
## Homogeneity:  
## Completeness: 
def entropy(counts, total):
  entropy = 0.0
  p = 0
  for count in counts:
    p = 1.0*counts[count]/total
    if p != 0.0:
      entropy -= p*math.log(p)/math.log(2)
  return entropy

def mutualInformation(clusters, tags, counts, total):
  MI = 0.0
  for cluster in clusters:
    cProb = 1.0*clusters[cluster]/total
    for tag in tags:
      tProb = 1.0*tags[tag]/total;
      coProb = 1.0*counts[tag][cluster]/total
      if coProb != 0:
        MI += coProb*math.log(coProb/(tProb*cProb))/math.log(2)
  return MI

clusterTotal = {}
goldTotal = {}
total = 0
for gold in C:
  for cluster in C[gold]:
    if cluster not in clusterTotal:
      clusterTotal[cluster] = 0
    if gold not in goldTotal:
      goldTotal[gold] = 0
    clusterTotal[cluster] += C[gold][cluster]
    goldTotal[gold] += C[gold][cluster]
    total += C[gold][cluster]
clusterEntropy = entropy(clusterTotal, total)
tagEntropy = entropy(goldTotal, total)
MI = mutualInformation(clusterTotal, goldTotal, C, total)
clusterGivenTag = clusterEntropy - MI
tagGivenCluster = tagEntropy - MI
c = 1 - (clusterGivenTag / clusterEntropy)
h = 1 - (tagGivenCluster / tagEntropy)


print "VM:  %6.3f  of %5.3f  %5.3f" % (100 * 2*h*c/(h+c),h,c)
print "VI:  %6.3f  of %5.3f  %5.3f" % (clusterGivenTag + tagGivenCluster, clusterGivenTag, tagGivenCluster)
