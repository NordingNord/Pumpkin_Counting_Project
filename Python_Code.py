# Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

# Constants
Blue = 0
Green = 1
Red = 2

# Exercise 3.1.1

# Orginial image
Image_Path = "Images/EB-02-660_0595_0007.JPG"
Original_Image = cv2.imread(Image_Path)

# Image used for annotation:
Annotation_Path = "Annotated_Image.png"
Anotation_Image = cv2.imread(Annotation_Path)

# Annotation to mask function
def Mask_From_Annotation(Anotated_Image,Anotation_Colour):
    # Initialize error variable
    Error = 0
    Low_Limit = (0,0,0)
    Up_Limit = (0,0,0)

    # Make limits based on annotation colours
    if Anotation_Colour == 0:
        Low_Limit = (255,0,0)
        Up_Limit = (255,0,0)
    elif Anotation_Colour == 1:
        Low_Limit = (0,255,0)
        Up_Limit = (0,255,0)
    elif Anotation_Colour == 2:
        Low_Limit = (0,0,255)
        Up_Limit = (0,0,255)
    else:
        print("Annotation colour not available")

    # Make mask
    if Error == 0:
        Mask = cv2.inRange(Anotated_Image, Low_Limit, Up_Limit)
        return Mask
    else:
        print("Mask making failed")

# Determine standard deviation and mean
def Get_Mask_Mean_STD(Original,The_Mask):
    Mean,std = cv2.meanStdDev(Original, mask = The_Mask)
    print("Found mean: (%f,%f,%f)" % (Mean[0,0], Mean[1,0], Mean[2,0]))
    print("Found Standard deviation (%f,%f,%f)" % (std[0,0],std[1,0],std[2,0]))
    New_Mean = (Mean[0,0],Mean[1,0],Mean[2,0])
    return New_Mean,std

# Using BGE
print("BGR colour space results:")
Annotation_Mask = Mask_From_Annotation(Anotation_Image,Red)
BGR_Mean,std = Get_Mask_Mean_STD(Original_Image,Annotation_Mask)

# Using CieLAB
print("CieLAB colour space results:")
CieLAB_Original = cv2.cvtColor(Original_Image, cv2.COLOR_BGR2Lab)
LAB_Mean,std = Get_Mask_Mean_STD(CieLAB_Original,Annotation_Mask)

# Get pixel values from mask
def Get_Annotated_Pixel_Values(Image,Mask):
    Pixels = np.reshape(Image,(-1,3))
    Mask_Pixels = np.reshape(Mask,(-1)) # -1 because only white or black pixels so we dont need 3 channels
    Annotated_Pixel_Values = Pixels[Mask_Pixels == 255] # Take only pixels where the mask is 255
    return Annotated_Pixel_Values

# Get average and covariance from annotated pixel values
def Get_Avg_Cov(Annotated_Values):
    avg = np.average(Annotated_Values, axis=0)
    cov = np.cov(Annotated_Values.transpose())
    print("Pixel value average: (%f,%f,%f)" %(avg[0],avg[1],avg[2]))
    return avg,cov

# Visualize pixel values
def Visualize_Pixel_Values(Pixels):
    fig,ax1 = plt.subplots()
    ax1.plot(Pixels[:,1],Pixels[:,2],'.')
    ax1.set_title('Pixel Colour values')
    plt.xlabel("Green [0-255]")
    plt.ylabel("Red [0-255]")
    fig.tight_layout()
    plt.savefig("Colour_Distribution.pdf",dpi=150)

# Average colour value
Pixel_Values = Get_Annotated_Pixel_Values(Original_Image,Annotation_Mask)
Average,Covariance = Get_Avg_Cov(Pixel_Values)

# Visualize pixel values
Visualize_Pixel_Values(Pixel_Values)

# Exercise 3.1.2

def Segmentation_From_Mean(Image,Mean,Thresholds_Up, Thresholds_Low):
    Low = (Mean[0]-Thresholds_Low[0],Mean[1]-Thresholds_Low[1],Mean[2]-Thresholds_Low[2])
    Up = (Mean[0]+Thresholds_Up[0],Mean[1]+Thresholds_Up[1],Mean[2]+Thresholds_Up[2])
    Segmented_Image = cv2.inRange(Image,Low,Up)
    return Segmented_Image

def Compare_Images(Original,Segmented,Title):
    # Convert to RGB since that is what pyplotlib uses
    RGB_Original = cv2.cvtColor(Original,cv2.COLOR_BGR2RGB)
    RGB_Segmented = cv2.cvtColor(Segmented,cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(9,3))
    ax1 = plt.subplot(1,2,1)
    plt.title(Title)
    ax1.imshow(RGB_Original)
    ax2 = plt.subplot(1,2,2,sharex=ax1,sharey=ax1)
    ax2.imshow(RGB_Segmented)
    plt.show()

# First the BGR segmentation
Thresholds_Up = (50,50,90)
Thresholds_Low = (10,10,20)
BGR_Segmented = Segmentation_From_Mean(Original_Image,BGR_Mean,Thresholds_Up, Thresholds_Low)
Compare_Images(Original_Image,BGR_Segmented,"BGR Segmentation result")

# Now with CieLAB
Thresholds_Up = (40,10,15)
Thresholds_Low = (80,15,25)
CieLAB_Segmented = Segmentation_From_Mean(CieLAB_Original,LAB_Mean,Thresholds_Up, Thresholds_Low)
Compare_Images(Original_Image,CieLAB_Segmented,"CieLAB Segmentation result")

# Now with distance to reference colour (squared Mahalanobis distance)
def Get_Squared_Mahalanobis(Image,Cov,Pixels,Reference_Colour,Dist_Thresh = 5):
    Inverse_Cov = np.linalg.inv(Cov)
    Diff = Pixels - Reference_Colour
    ModDotProduct = Diff * (Diff @ Inverse_Cov) # @ = matrix multiplication, * means pointwise multiplication
    Mahalanobis_Dist = np.sum(ModDotProduct,axis=1)
    # Only keep pixels close enought to the colour orange
    Threshold_Mahalanobis = cv2.inRange(Mahalanobis_Dist,0,Dist_Thresh)
    Mahalanobis_Distance_Image = np.reshape(Threshold_Mahalanobis,(Image.shape[0],Image.shape[1]))
    return  Mahalanobis_Dist,Mahalanobis_Distance_Image

# Found covariance of pixels earlier
#Reference_Colour = (0,165,255) # Orange
Reference_Colour = (99,181,255) # Shiny orange from image
Image_Pixels = np.reshape(Original_Image,(-1,3))
Mahalanobis_Distances,Mahalanobis_Image = Get_Squared_Mahalanobis(Original_Image,Covariance,Image_Pixels,Reference_Colour)
Compare_Images(Original_Image,Mahalanobis_Image, "Mahalanobis implementation")

# 3.1.3: I Choose Mahalanobis

# 3.2.1: Count number of orange blobs in the segmented image
def Count_Objects(Image, Offset = (0,0)):
    # I use find contours
    Contours,Hierarchy = cv2.findContours(Image,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,offset=Offset) # Contour method and approximation method
    print("There are %i pumpkins in the image" %len(Contours))
    return Contours

Contours = Count_Objects(Mahalanobis_Image)

# 3.2.2: Filter segmented image to remove noise

def Remove_Noise(Image):
    # Median filter (find median in every 3x3 and set the middle pixel to that)
    Median_Image = cv2.medianBlur(Image,3)
    # Compare with previous
    Compare_Images(Image,Median_Image, "Noise reduction")
    return Median_Image

Filtered_Segmented_Image = Remove_Noise(Mahalanobis_Image)

# 3.2.3: Count orange blobs with filtered image
Filtered_Contours = Count_Objects(Filtered_Segmented_Image)
# Resulted in fewer pumpkins 6011 -> 5190, lets hope its only noise pumpkins removed

# 3.2.4: Mark located pumpkins in the input image
def Mark_Objects(Image,Contours,Offset_x = 0, Offset_y = 0, Compare = 1):
    Mark_Image = Image.copy()
    cv2.drawContours(Mark_Image,Contours,-1,(255,0,0),2,offset = (Offset_x,Offset_y)) # ContourIdx, Colour, thickness
    if Compare == 1:
        Compare_Images(Image,Mark_Image, "Found pumpkins")
    return Mark_Image

Marked_Image = Mark_Objects(Original_Image,Contours)

# 3.4.1 Load part of orthomosaic
Orthomosaic_Name = "Cut_Field.tif"

def Get_Orthomosaic_Resolution(File_Name):
    with rasterio.open(File_Name) as src:
        Cols = src.width
        Rows = src.height
    return Cols,Rows

def Load_Part_Orthomosaic(File_Name,Row_Start, Row_End, Col_Start, Col_End):
    with rasterio.open(File_Name) as src:
        # ulc stands for upper left cornor
        #lrc stands for lower right cornor and so on
        window_location = Window.from_slices((Row_Start,Row_End),(Col_Start,Col_End))
        #print(window_location)
        Orthomosaic_Part = src.read(window=window_location)

        # Fix image shape
        Temp = Orthomosaic_Part.transpose(1,2,0)
        t2 = cv2.split(Temp)
        Orthomosaic_Part_cv = cv2.merge([t2[2],t2[1],t2[0]])

    return Orthomosaic_Part_cv

# Show part of orthomosaic
Cols, Rows = Get_Orthomosaic_Resolution(Orthomosaic_Name)
print("Cols: %i Rows: %i" %(Cols,Rows))
Col_Start = 0
Row_Start = 0 # Just take upper left corner
Col_End = 10410
Row_End = 2367
Part_Orthomosaic = Load_Part_Orthomosaic(Orthomosaic_Name,Row_Start,Row_End,Col_Start,Col_End)
Orthomosaic_RGB = cv2.cvtColor(Part_Orthomosaic,cv2.COLOR_BGR2RGB)
plt.imshow(Orthomosaic_RGB)
plt.show()
# Black because a black void surounds the map, since we changed its shape

# 3.4.2 Design tile placement including overlaps

# I decide tile sizes based on the orthomosaic resolution
# There are 10410 columns. I choose to divide these by 30 to get 30 column tiles of 347 column length
Tile_Col_Size = Cols/30
print("Tile Column size: %f" %Tile_Col_Size)

# There are 2367 rows. I divide with 9 to get 9 row tiles with 263 row length
Tile_Row_Size = Rows/9
print("Tile Row size: %f" %Tile_Row_Size)

# I will deal with edge pumpkins by deleting half of them, since they would be counted twice (same pumpkin on two tiles)

# 3.4.4 Deal with pumpkins in the overlap
# I create a function that will take amount of pumpkins on edge and minus the total with half of that
def Count_Overlappers(Object_Image,Image):
    test = Image.copy()
    Rows,Cols = Object_Image.shape
    White = 255
    Black = 0
    Edge_Pumpkins = 0
    # Go Through top row
    i = 0
    while i < Cols:
        pixel = Object_Image[0,i]
        if pixel == White:
            # Edge pumpkin found
            Edge_Pumpkins = Edge_Pumpkins+1
            test[0,i] = (0,0,255)
            # Continue until no more white
            while pixel == White:
                if(i == Cols-1):
                    break
                else:
                    i = i+1
                pixel = Object_Image[0,i]
                if(pixel == Black and i+1 < Cols-1):
                    pixel = Object_Image[0,i+1]
                    if(pixel == Black and i+2 < Cols-1):
                        pixel = Object_Image[0,i+2]
                    
        i = i+1
    # Bottom Row
    i = 0
    while i < Cols:
        pixel = Object_Image[Rows-1,i]
        if pixel == White:
            # Edge pumpkin found
            Edge_Pumpkins = Edge_Pumpkins+1
            test[Rows-1,i] = (0,0,255)
            # Continue until no more white
            while pixel == White:
                if(i == Cols-1):
                    break
                else:
                    i = i+1
                pixel = Object_Image[Rows-1,i]
                if(pixel == Black and i+1 < Cols-1):
                    pixel = Object_Image[Rows-1,i+1]
                    if(pixel == Black and i+2 < Cols-1):
                        pixel = Object_Image[Rows-1,i+2]
        i = i+1
    # First column
    i = 0
    while i < Rows:
        pixel = Object_Image[i,0]
        if pixel == White:
            # Edge pumpkin found
            Edge_Pumpkins = Edge_Pumpkins+1
            test[i,0] = (0,0,255)
            # Continue until no more white
            while pixel == White:
                if(i == Rows-1):
                    break
                else:
                    i = i+1
                pixel = Object_Image[i,0]
                if(pixel == Black and i+1 < Rows-1):
                    pixel = Object_Image[i+1,0] 
                    if(pixel == Black and i+2 < Rows-1):
                        pixel = Object_Image[i+2,0]
        i = i+1
    # Last column
    i = 0
    while i < Rows:
        pixel = Object_Image[i,Cols-1]
        if pixel == White:
            # Edge pumpkin found
            Edge_Pumpkins = Edge_Pumpkins+1
            test[i,Cols-1] = (0,0,255)
            # Continue until no more white
            while pixel == White:
                if(i == Rows-1):
                    break
                else:
                    i = i+1
                pixel = Object_Image[i,Cols-1]
                if(pixel == Black and i+1 < Rows-1):
                    pixel = Object_Image[i+1,Cols-1]
                    if(pixel == Black and i+2 < Rows-1):
                        pixel = Object_Image[i+2,Cols-1]
        i = i+1
    #Compare_Images(Object_Image,test,"Test")
    print("%i pumpkins found on the edge" %Edge_Pumpkins)
    return Edge_Pumpkins

# Function added to 3.4.3

# 3.4.3 Count pumpkins on each tile

def Count_Tile_Wise(Orthomosaic_Name,Col_Size,Row_Size,Ref,Cov,Overlap = 1):
    Col_Start = Row_Start = 0
    Col_End,Row_End = Col_Size,Row_Size
    Cols,Rows = Get_Orthomosaic_Resolution(Orthomosaic_Name)
    Sum_Pumpkins = 0
    # Get full image for drawing
    Col_Start_Full = 0
    Row_Start_Full = 0
    Col_End_Full = Cols
    Row_End_Full = Rows
    Full_Image = Load_Part_Orthomosaic(Orthomosaic_Name,Row_Start_Full,Row_End_Full,Col_Start_Full,Col_End_Full)
    # Go through each tile
    for Col in range(int(Cols/Col_Size)):
        for Row in range(int(Rows/Row_Size)):
            Offset_x = int(Col_Size*Col+Col)
            Offset_y = int(Row_Size*Row+Row) # Why is it offset weird
            # Get part of orthomosaic
            Working_Image = Load_Part_Orthomosaic(Orthomosaic_Name,Row_Start,Row_End,Col_Start,Col_End)
            print("New tile:")
            # Convert to mahalanobis segmented image
            Pixels = np.reshape(Working_Image,(-1,3))
            Mahalanobis_Distances,Mahalanobis_Image = Get_Squared_Mahalanobis(Working_Image,Cov,Pixels,Ref,6)
            # Conduct pumpkin counting
            Pumpkins = Count_Objects(Mahalanobis_Image)
            # Draw pumpkins
            Full_Image = Mark_Objects(Full_Image,Pumpkins,Offset_x,Offset_y,0)
            if Overlap == 1:
                # Find number of edge pumpkins
                Overlappers = Count_Overlappers(Mahalanobis_Image, Working_Image)
                Sum_Pumpkins = Sum_Pumpkins+len(Pumpkins)-int(Overlappers/2)
            else:
                Sum_Pumpkins = Sum_Pumpkins+len(Pumpkins)
            #Part_Marked = Mark_Objects(Working_Image,Pumpkins,0,0,1)
            # Update row
            Row_Start,Row_End = Row_End+1,Row_End+Row_Size+1
        # Update Col
        Col_Start, Col_End = Col_End+1,Col_End+Col_Size+1
        # Reset Row
        Row_Start,Row_End = 0,Row_Size
    # Final count
    print("All tiles searched")
    print("Total amount of pumpkins: %i" %Sum_Pumpkins)
    return Full_Image

Pumkin_Image = Count_Tile_Wise(Orthomosaic_Name,Tile_Col_Size,Tile_Row_Size,Average,Covariance,0)
Orthomosaic_BGR = cv2.cvtColor(Orthomosaic_RGB,cv2.COLOR_RGB2BGR)
Compare_Images(Orthomosaic_BGR,Pumkin_Image,"Orthomosaic pumpkins")


# 3.4.5 Determine pumpkins on entire field
# Already implemented in 3.4.3

# Count all non black pixels
Col_Start = 0
Row_Start = 0 # Just take upper left corner
Col_End = 10410
Row_End = 2367
Full_Orthomosaic = Load_Part_Orthomosaic(Orthomosaic_Name,Row_Start,Row_End,Col_Start,Col_End)

pix = 0
i = 0
j = 0
while i < Cols:
    while j < Rows:
        if (Full_Orthomosaic[j,i,0] != 0 and Full_Orthomosaic[j,i,1] != 0 and Full_Orthomosaic[j,i,0] != 2):
            pix = pix+1
        j = j+1
    i = i+1
    j = 0
print(pix)






