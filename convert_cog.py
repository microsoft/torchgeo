import os

dirpath = "/projects/dali/data/"
sourcedir = "l7irish"
targetdirs = ["l7irish_cog", "l7irish_trans", "l7irish_trans2"]

os.chdir(dirpath)
for targetdir in targetdirs:
    os.system("rsync -a --include='*/' --exclude='*' " + sourcedir + "/ " + targetdir)
    top = dirpath + sourcedir

    for root, dirs, files in os.walk(top, topdown=False):
        for file in files:
            fp = os.path.join(root, file)
            if fp.endswith(".TIF"):
                print("Processing: ", fp)
                fp_cog = fp.replace(sourcedir, targetdir)
                if targetdir == "l7irish_cog":
                    os.system("gdalwarp -of COG -co COMPRESS=LZW -co PREDICTOR=2 -q " + fp + " " + fp_cog)
                elif targetdir == "l7irish_trans":
                    os.system("gdal_translate -co TILED=YES -q " + fp + " " + fp_cog)
                else:
                    os.system("gdal_translate -co TILED=YES -co COPY_SRC_OVERVIEWS=YES -q " + fp + " " + fp_cog)
    print("complete " + targetdir)
print("done")