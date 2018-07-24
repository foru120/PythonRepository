def random_crop():
    c_ims = dict()
    for root, dirs, files in os.walk("./png"):
        for file in files:
            imagePath = os.path.join(root, file)
            im = Image.open(imagePath)
            w = 224
            h = 224
            rx = np.random.randint(0, im.size[0] - w)
            ry = np.random.randint(0, im.size[1] - h)
            pre=imagePath[:-4]
            index=pre.split("\\")
            if index[1] not in c_ims:
                c_ims[index[1]]=dict()
            patient=c_ims[index[1]]
            if index[2] not in patient:
                patient[index[2]]=dict()
            study=patient[index[2]]
            if index[3] not in study:
                study[index[3]]=dict()
            series=study[index[3]]
            if index[4] not in series:
                series[index[4]]=dict()
            instance=series[index[4]]
            if index[5] not in instance:
                instance[index[5]]=dict()
            instance[index[5]]=im.crop((rx, ry, rx + w, ry + h))
    return c_ims