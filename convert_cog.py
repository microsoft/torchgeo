import os

dirpath = "/projects/dali/data/"
sourcedir = "l7irish"
# targetdirs = ["l7irish_cog", "l7irish_trans", "l7irish_trans2", "l7irish_3857"]
targetdirs = ["l7irish_cog_3857"]
targetbiomes = ["mid_latitude_south", "polar_north"]

os.chdir(dirpath)
for targetdir in targetdirs:
    os.system("rsync -a --include='*/' --exclude='*' " + sourcedir + "/mid_latitude_south/ " + targetdir + "/mid_latitude_south/")
    os.system("rsync -a --include='*/' --exclude='*' " + sourcedir + "/polar_north/ " + targetdir + "/polar_north/")

    for targetbiome in targetbiomes:
        top = dirpath + sourcedir + "/" + targetbiome
        for root, dirs, files in os.walk(top, topdown=False):
            for file in files:
                fp = os.path.join(root, file)
                if fp.endswith(".TIF"):
                    print("Processing: ", fp)
                    fp_cog = fp.replace(sourcedir, targetdir)
                    os.system("gdalwarp -of COG -co COMPRESS=LZW -co PREDICTOR=2 -q -t_srs EPSG:3857 " + fp + " " + fp_cog)
                    #if targetdir == "l7irish_cog":
                    #    os.system("gdalwarp -of COG -co COMPRESS=LZW -co PREDICTOR=2 -q " + fp + " " + fp_cog)
                    #elif targetdir == "l7irish_trans":
                    #    os.system("gdal_translate -co TILED=YES -q " + fp + " " + fp_cog)
                    #else:
                    #    os.system("gdal_translate -co TILED=YES -co COPY_SRC_OVERVIEWS=YES -q " + fp + " " + fp_cog)
    print("complete " + targetdir)
print("done")
