import numpy as np
import os
from visualize.vis_samples import flatten
from utils.Utils import args




def initDataFromDictionary(data_path, years):
    train_list = []
    assert data_path
    for i in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, i)):
            ip = data_path + i + '/'

            for j in os.listdir(ip):
                jp = ip + j
                if years and int(i) in years:
                    train_list.append(jp)

    return train_list


def getOneTyphoon(dir):
    mvts = []
    files = sorted([os.path.join(dir, i) for i in os.listdir(dir)])
    channel1 = sorted(os.listdir(files[0]))

    for index in range(len(channel1)):
        image = channel1[index]
        if image.endswith('jpg'):
            temp = image.split('-')
            ori_intense = float(temp[-1].split('.')[0])
            if str(temp[1]) not in args.time_spot or ori_intense==0:
                continue
            mvts.append(ori_intense)
    return mvts


def get_cma_track(path='F:/data/TC_IR_IMAGE/2000-2019.txt'):
    f = open(path)
    tc_record = {}
    one_record = []
    id = ''
    lines = f.readlines()
    for index in range(len(lines)):
        line = lines[index]
        if line.strip() == '':
            continue
        content = list(filter(None, line.split(" ")))
        if index == (len(lines) - 1):
            if len(one_record) > 0 and 'nameless' not in id:
                tc_record[id] = one_record
        if content[0] == '66666':
            if len(one_record) > 0 and 'nameless' not in id:
                tc_record[id] = one_record

            year = lines[index + 1].split(' ')[0][0:4]
            name = content[7].lower()
            id = '_'.join([year, name]).strip()
            one_record = []
        else:
            one_record.append(content)
    f.close()
    return tc_record


def save_labels(data_file='F:/data/TC_IR_IMAGE/'):
    labels = {}
    tys = initDataFromDictionary(data_file, args.train_years)
    for path in tys:
        basename = os.path.basename(path)
        number = basename.split('_')[0]
        intensities = getOneTyphoon(path)
        labels[number] = intensities

    np.save('F:/Python_Project/TC_intensity_prediction/Plots/label_ints.npy', labels, allow_pickle=True)
    labels = flatten(labels)
    print(len(labels))
    return labels
# save_labels()

def match_one_ty(cma, jma):
    cma_record = {}
    for i in cma:
        cma_record[i[0]] = i[2:]
    for image in jma:
        temp = os.path.basename(image).split('-')
        pre_name='-'.join(temp[:-1])
        ori_intense = float(temp[-1].split('.')[0])
        hour = str(temp[1])
        date = temp[0]
        if hour not in args.time_spot:
            continue
        id = date + hour[0:2]
        if id not in cma_record.keys():
            if ori_intense != 0:
                print('label not found',image, id)
        else:
            new_ints=cma_record[id][-1].strip()+'.jpg'
            new_name='-'.join([pre_name,new_ints])
            new_name=os.path.join(os.path.dirname(image),new_name)
            print(image,'======>',new_name)
            os.rename(image,new_name)


def intensity_match(cma_track, data_file='F:/data/TC_IR_IMAGE/'):
    tys = initDataFromDictionary(data_file, args.train_years)
    for path in tys:
        basename = os.path.basename(path)
        id = basename.split('_')
        year = id[0][:4]
        name = id[1].lower()
        id = '_'.join([year, name])
        if id not in cma_track.keys():
            print(id)
        else:
            names = []
            files = sorted([os.path.join(path, i) for i in os.listdir(path)])
            channel1 = sorted(os.listdir(files[0]))

            for index in range(len(channel1)):
                image = channel1[index]
                if image.endswith('jpg'):
                    names.append(os.path.join(files[0], image))
            match_one_ty(cma_track[id], names)
        # return

if __name__ == '__main__':

    intensity_match(cma_track=get_cma_track())
