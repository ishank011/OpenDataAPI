#Code to extract features and write them to mortality.data

f1 = open('static/data2/id_age_train.csv', 'r')
reader = csv.reader(f1)
data = [row for row in reader]
data.pop(0)
del reader
f1.close()
train_data = np.array([list(map(int, a)) for a in data])
#print train_data[0], train_data[1]
del data

f1 = open('static/data2/id_label_train.csv', 'r')
reader = csv.reader(f1)
data = [row for row in reader]
data.pop(0)
del reader
f1.close()
train_labels = [list(map(int, a)) for a in data]
train_labels = np.delete(train_labels, 0, 1)
#print train_labels[0], train_labels[1]
del data

f1 = open('static/data2/id_time_vitals_train.csv', 'r')
reader = csv.reader(f1)
data = [row for row in reader]
data.pop(0)
del reader
f1.close()
for j in range(len(data)):
    for i in range(len(data[j])):
    	if data[j][i] == 'NA':
    		data[j][i] = np.nan
    	else:
    		data[j][i] = float(data[j][i])
vitals = np.array(data)
del data

features = []
for i in range(train_data.shape[0]):
    a = vitals[vitals[:, 0] == train_data[i][0]]
    #print len(a)
    p1 = np.nanmean(a[:, 2:8], axis=0)
    p2 = np.nanmax(a[:, 2:8], axis=0)
    p3 = np.nanmin(a[:, 2:8], axis=0)
    p4 = np.nanstd(a[:, 2:8], axis=0)
    p5 = p2 - p3
    features.append(np.concatenate((p1, p2, p3, p4, p5), axis=1))

train_data = np.concatenate((train_data, np.array(features)), axis=1)
train_data = np.delete(train_data, 0, 1)
del features

imp = Imputer(copy=False)
imp.fit_transform(train_data)
print train_data.shape

f1 = open('static/data2/mortality.data', 'w')
for i in range(train_data.shape[0]):
	f1.write(','.join([str(a) for a in train_data[i]])+','+str(train_labels[i][0])+'\n')
f1.close()