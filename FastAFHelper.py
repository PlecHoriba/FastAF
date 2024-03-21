import sys
from ctypes import*
from xmlrpc.client import DateTime # import ctypes (used to call DLL functions)
import ids_peak.ids_peak as ids_peak
import ids_peak_ipl.ids_peak_ipl as ids_ipl
import numpy as np
import time
import cv2
import skimage
import math
import easygui

m_Tango = cdll.LoadLibrary("D:\\Users\\leclerc\\Documents\\Script\\LRS Fast AF\\TangoDLL_64bit_V1414\\Tango_DLL.dll") # give location of dll (current directory)("TangoDLL_64bit_V1414\\Tango_DLL.dll")   

class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = DateTime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (DateTime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time

# Tango :
    
def init_tango() :
    ####################################################
    ## initialize Tango, the electronic of MarzHauser ##
    ####################################################
    if m_Tango == 0:
        print("Error: failed to load DLL")
        sys.exit(0)
        
    # Tango_DLL.dll loaded successfully

    if m_Tango.LSX_CreateLSID == 0:
        print("unexpected error. required DLL function CreateLSID() missing")
        sys.exit(0)
    # continue only if required function exists

    LSID = c_int()
    error = int     #value is either DLL or Tango error number if not zero
    error = m_Tango.LSX_CreateLSID(byref(LSID))
    if error > 0:
        print("Error: " + str(error))
        sys.exit(0)
        
    # OK: got communication ID from DLL (usually 1. may vary with multiple connections)
    # keep this LSID in mind during the whole session
        
    if m_Tango.LSX_ConnectSimple == 0:
        print("unexepcted error. required DLL function ConnectSimple() missing")
        sys.exit(0)
    # continue only if required function exists

    # error = m_Tango.LSX_ConnectSimple(LSID,2,"COM20",57600,0)
    # following combination of -1,"" works only for USB but not for RS232 connections 
    error = m_Tango.LSX_ConnectSimple(LSID,-1,"",57600,0)
    if error > 0:
        print("Error: LSX_ConnectSimple " + str(error))
        sys.exit(0)

    print("TANGO is now successfully connected to DLL")


    # some c-type variables (general purpose usage)
    dx = c_double()
    dy = c_double()
    dz = c_double()
    da = c_double()

    # query actual position (4 axes) (unit depends on GetDimensions)
    error = m_Tango.LSX_GetPos(LSID,byref(dx),byref(dy),byref(dz),byref(da))
    if error > 0:
        print("Error: GetPos " + str(error))
    else:
        print("position = " + str(dx.value) + " " + str(dy.value) + " " + str(dz.value) + " " + str(da.value))

    # query actual axes accelerations (unit is m/s�)
    error = m_Tango.LSX_GetAccel(LSID,byref(dx),byref(dy),byref(dz),byref(da))
    if error > 0:
        print("Error: GetAccel " + str(error))
    else:
        print("acceleration = " + str(dx.value) + " " + str(dy.value) + " " + str(dz.value) + " " + str(da.value))


    # query pitch factor
    error = m_Tango.LSX_GetPitch(LSID,byref(dx),byref(dy),byref(dz),byref(da))
    if error > 0:
        print("Error: GetPitch " + str(error))
    else:
        print("pitch = " + str(dx.value) + " " + str(dy.value) + " " + str(dz.value) + " " + str(da.value))


    # query gear factor
    error = m_Tango.LSX_GetGear(LSID,byref(dx),byref(dy),byref(dz),byref(da))
    if error > 0:
        print("Error: GetGear " + str(error))
    else:
        print("gear = " + str(dx.value) + " " + str(dy.value) + " " + str(dz.value) + " " + str(da.value))


    # query motor current (in A)
    error = m_Tango.LSX_GetMotorCurrent(LSID,byref(dx),byref(dy),byref(dz),byref(da))
    if error > 0:
        print("Error: GetMotorCurrent " + str(error))
    else:
        print("motor current = " + str(dx.value) + " " + str(dy.value) + " " + str(dz.value) + " " + str(da.value))

    return m_Tango, LSID

def close_tango(m_Tango,LSID):
    error = m_Tango.LSX_Disconnect(LSID)
    if error > 0:
        print("Error: LSX_Disconnect " + str(error))
        sys.exit(0)

    print("TANGO is now successfully disconnected to DLL")

    error = m_Tango.LSX_FreeLSID(LSID)
    if error > 0:
        print("Error: " + str(error))
        sys.exit(0)

    print("lsid freed")

def moveRelSingleAxistango(LSID, Axis, qte):
    Mvt =c_double(qte)
    error = m_Tango.LSX_MoveRelSingleAxis(LSID, Axis, Mvt, True)
    if error > 0:
        print("Error: GetMotorCurrent " + str(error))
    else:
        dx = c_double()
        dy = c_double()
        dz = c_double()
        da = c_double()
        error = m_Tango.LSX_GetPos(LSID,byref(dx),byref(dy),byref(dz),byref(da))
        print("position  = " + str(dx.value) + " " + str(dy.value) + " " + str(dz.value) + " " + str(da.value))

def moveAbsSingleAxistango(LSID, Axis, qte):
    Mvt =c_double(qte)
    error = m_Tango.LSX_MoveAbsSingleAxis(LSID, Axis, Mvt, True)
    if error > 0:
        print("Error: GetMotorCurrent " + str(error))
    else:
        dx = c_double()
        dy = c_double()
        dz = c_double()
        da = c_double()
        error = m_Tango.LSX_GetPos(LSID,byref(dx),byref(dy),byref(dz),byref(da))
        print("position  = " + str(dx.value) + " " + str(dy.value) + " " + str(dz.value) + " " + str(da.value))

def TangoGetPosSingleAxis(LSID, Axis):
    # query actual position (4 axes) (unit depends on GetDimensions)
    dz = c_double()
    error = m_Tango.LSX_GetPosSingleAxis(LSID,Axis,byref(dz))
    if error > 0:
        print("Error: GetPos " + str(error))
    else:
        # print("position = " + str(dz.value))
        pass
    return error, dz

# Camera :
        
def set_roi_camera(x, y, width, height, remote_device_nodemap):
    # Get the minimum ROI and set it. After that there are no size restrictions anymore
    x_min = remote_device_nodemap.FindNode("OffsetX").Minimum()
    y_min = remote_device_nodemap.FindNode("OffsetY").Minimum()
    w_min = remote_device_nodemap.FindNode("Width").Minimum()
    h_min = remote_device_nodemap.FindNode("Height").Minimum()

    remote_device_nodemap.FindNode("OffsetX").SetValue(x_min)
    remote_device_nodemap.FindNode("OffsetY").SetValue(y_min)
    remote_device_nodemap.FindNode("Width").SetValue(w_min)
    remote_device_nodemap.FindNode("Height").SetValue(h_min)

    # Get the maximum ROI values
    x_max = remote_device_nodemap.FindNode("OffsetX").Maximum()
    y_max = remote_device_nodemap.FindNode("OffsetY").Maximum()
    w_max = remote_device_nodemap.FindNode("Width").Maximum()
    h_max = remote_device_nodemap.FindNode("Height").Maximum()

    if (x < x_min) or (y < y_min) or (x > x_max) or (y > y_max):
        return False
    elif (width < w_min) or (height < h_min) or ((x + width) > w_max) or ((y + height) > h_max):
        return False
    else:
        # Now, set final AOI
        remote_device_nodemap.FindNode("OffsetX").SetValue(x)
        remote_device_nodemap.FindNode("OffsetY").SetValue(y)
        remote_device_nodemap.FindNode("Width").SetValue(width)
        remote_device_nodemap.FindNode("Height").SetValue(height)
        print ('ROI Define Done!')
        return True
    
def init_camera(ValueExpoTime):
    device = None
    datastream = None
    remote_device_nodemap = None

    ## initialize library and create camera instance
    ids_peak.Library.Initialize()
    device_manager = ids_peak.DeviceManager.Instance()
    device_manager.Update()
    device_descriptors = device_manager.Devices()

    print("Found Devices: " + str(len(device_descriptors)))
    for device_descriptor in device_descriptors:
        print(device_descriptor.DisplayName())
        
    device = device_descriptors[0].OpenDevice(ids_peak.DeviceAccessType_Exclusive)
    print("Opened Device: " + device.DisplayName())
    remote_device_nodemap = device.RemoteDevice().NodeMaps()[0]

    remote_device_nodemap.FindNode("AcquisitionMode").SetCurrentEntry("Continuous")
    remote_device_nodemap.FindNode("TriggerSelector").SetCurrentEntry("ExposureStart")
    remote_device_nodemap.FindNode("TriggerSource").SetCurrentEntry("Software")
    remote_device_nodemap.FindNode("TriggerMode").SetCurrentEntry("Off") # should be switch to on to stop freerun
                
    # Get the current frame rate
    frame_rate = remote_device_nodemap.FindNode("AcquisitionFrameRate").Value()
    print("the framerate is ", frame_rate, "frame per sec")      

    ## start datastream
    datastream = device.DataStreams()[0].OpenDataStream()
    datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
 
    # Clear all old buffers
    for buffer in datastream.AnnouncedBuffers():
        datastream.RevokeBuffer(buffer)

    payload_size = remote_device_nodemap.FindNode("PayloadSize").Value()

    # Alloc buffer
    for i in range(datastream.NumBuffersAnnouncedMinRequired()):
        buffer = datastream.AllocAndAnnounceBuffer(payload_size)
        datastream.QueueBuffer(buffer)
    
    datastream.StartAcquisition()
    remote_device_nodemap.FindNode("AcquisitionStart").Execute()
    remote_device_nodemap.FindNode("AcquisitionStart").WaitUntilDone()

    remote_device_nodemap.FindNode("ExposureTime").SetValue(ValueExpoTime) # in microseconds
    remote_device_nodemap.FindNode("AcquisitionFrameRate").SetValue(50)
    remote_device_nodemap.FindNode("ReverseX").SetValue(False)
    remote_device_nodemap.FindNode("ReverseY").SetValue(False)

    return remote_device_nodemap, datastream,device

def setExpoTime_camera(remote_device_nodemap,ValueExpoTime):
    remote_device_nodemap.FindNode("ExposureTime").SetValue(ValueExpoTime)

def close_camera():
    ids_peak.Library.Close()


## Acquisition (make the program bug for unknown reasons) 
def camera_takeoneimage(remote_device_nodemap,datastream):
    ## This function makes the program bug, for some reason the buffer does not play nice when beeing called
    # trigger image
    remote_device_nodemap.FindNode("TriggerSoftware").Execute()
    buffer = datastream.WaitForFinishedBuffer(1000)

    # convert to RGB
    raw_image = ids_ipl.Image.CreateFromSizeAndBuffer(buffer.PixelFormat(), buffer.BasePtr(), buffer.Size(), buffer.Width(), buffer.Height())
    color_image = raw_image.ConvertTo(ids_ipl.PixelFormatName_RGB8)
    datastream.QueueBuffer(buffer)

    picture = color_image.get_numpy_3D()
    return picture
## Non utilisée takemovie ne sert pas a grand chose
def camera_takeonemovie(remote_device_nodemap,datastream):
    while (True): 
        # trigger image
        remote_device_nodemap.FindNode("TriggerSoftware").Execute()
        buffer = datastream.WaitForFinishedBuffer(100)

        # convert to RGB
        raw_image = ids_ipl.Image.CreateFromSizeAndBuffer(buffer.PixelFormat(), buffer.BasePtr(), buffer.Size(), buffer.Width(), buffer.Height())
        color_image = raw_image.ConvertTo(ids_ipl.PixelFormatName_RGB8)
        datastream.QueueBuffer(buffer)

        picture = color_image.get_numpy_3D()

        # putting the FPS count on the frame
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time
        # converting the fps into integer 
        fps = int(fps)
        fps = str(fps)
        
        # Display the resulting frame
        cv2.putText(picture, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('custom window', picture)
        h, w = picture.shape[0:2]
        neww = 800
        newh = int(neww*(h/w))
        cv2.resizeWindow('custom window', neww, newh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Destroy all the windows 
    cv2.destroyAllWindows()

# FASTAF :

## primary image analysis function  
def fincontours(img, Traitement, AutoSet ,kernel_size, Tresh = [250,255]):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    

    if AutoSet == True:
        # Test d'automatisation de fixationn des paramètres
        _,kernel_size = TestGranuImage(img_gray)
        Tresh = fixTreshvalue (img_gray)

    match Traitement:
        case 'erosion':
            # def kernel for erosion / dilatation
            kernel_dilate = np.ones((kernel_size,kernel_size),np.uint8)
            kernel_erode  = np.ones((kernel_size+2,kernel_size+2),np.uint8)
            
            # Thresholding image
            _,thresh   = cv2.threshold(img_gray,Tresh[0],Tresh[1],cv2.THRESH_BINARY)

            #erosion dilatation
            dilate      = cv2.dilate(thresh,kernel_dilate,iterations = 1)
            erosion_dst = cv2.erode(dilate,kernel_erode,iterations =1)

            # Contours
            erosion_dst = erosion_dst.astype(np.uint8)
            contours,_ = cv2.findContours(erosion_dst,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            return contours,thresh
        
        case 'erosion_Tresh_OTSU':
            # def kernel for erosion / dilatation
            kernel_dilate = np.ones((kernel_size,kernel_size),np.uint8)
            kernel_erode  = np.ones((kernel_size+2,kernel_size+2),np.uint8)
            
            # Thresholding image
            _,thresh   = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            #erosion dilatation
            dilate      = cv2.dilate(thresh,kernel_dilate,iterations = 1)
            erosion_dst = cv2.erode(dilate,kernel_erode,iterations =1)

            # Contours
            erosion_dst = erosion_dst.astype(np.uint8)
            contours,_ = cv2.findContours(erosion_dst,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            return contours,thresh

        case 'erosion_Tresh_Triangle':
            # def kernel for erosion / dilatation
            kernel_dilate = np.ones((kernel_size,kernel_size),np.uint8)
            kernel_erode  = np.ones((kernel_size+2,kernel_size+2),np.uint8)
            
            # Thresholding image
            # Compute the histogram
            from scipy.signal import find_peaks
            hist, bins = np.histogram(img_gray.flatten(), 256, [0, 256])

            # Find the peak of the histogram
            peaks, _ = find_peaks(hist)

            # Calculate Triangle threshold
            triangle_threshold = (peaks[0] + np.argmax(hist[peaks[0]:])) // 2

            # Apply Triangle thresholding
            _, thresh = cv2.threshold(img_gray, triangle_threshold, 255, cv2.THRESH_BINARY)

            #erosion dilatation
            dilate      = cv2.dilate(thresh,kernel_dilate,iterations = 1)
            erosion_dst = cv2.erode(dilate,kernel_erode,iterations =1)

            # Contours
            erosion_dst = erosion_dst.astype(np.uint8)
            contours,_ = cv2.findContours(erosion_dst,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            return contours,thresh
        
        case 'erosion_Tresh_Gauss':
            # def kernel for erosion / dilatation
            kernel_dilate = np.ones((kernel_size,kernel_size),np.uint8)
            kernel_erode  = np.ones((kernel_size+2,kernel_size+2),np.uint8)
            
            # Thresholding image
            thresh   = cv2.adaptiveThreshold(img_gray,Tresh[1],cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,-Tresh[0])

            #erosion dilatation
            dilate      = cv2.dilate(thresh,kernel_dilate,iterations = 1)
            erosion_dst = cv2.erode(dilate,kernel_erode,iterations =1)

            # Contours
            erosion_dst = erosion_dst.astype(np.uint8)
            contours,_ = cv2.findContours(erosion_dst,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            return contours,thresh
        
        case 'erosion_Tresh_Mean':
            # def kernel for erosion / dilatation
            kernel_dilate = np.ones((kernel_size,kernel_size),np.uint8)
            kernel_erode  = np.ones((kernel_size+2,kernel_size+2),np.uint8)
            
            # Thresholding image
            thresh   = cv2.adaptiveThreshold(img_gray,Tresh[1],cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,-Tresh[0])

            #erosion dilatation
            dilate      = cv2.dilate(thresh,kernel_dilate,iterations = 1)
            erosion_dst = cv2.erode(dilate,kernel_erode,iterations =1)

            # Contours
            erosion_dst = erosion_dst.astype(np.uint8)
            contours,_ = cv2.findContours(erosion_dst,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            return contours,thresh
        
        case 'butterworth':
            # def kernel for erosion / dilatation
            kernel_dilate = np.ones((kernel_size,kernel_size),np.uint8)
            kernel_erode  = np.ones((kernel_size+5,kernel_size+5),np.uint8)

            # Thresholding image
            _ , img_gray_smooth_treshed = cv2.threshold(img_gray,Tresh[0],Tresh[1],cv2.THRESH_TOZERO_INV)
            _ , img_gray_smooth_treshed = cv2.threshold(img_gray,Tresh[0]+50,Tresh[1],cv2.THRESH_BINARY)

            #erosion dilatation
            dilatation_dst = cv2.dilate(img_gray_smooth_treshed,kernel_dilate,iterations = 1)
            Filtered_image = skimage.filters.butterworth(dilatation_dst, cutoff_frequency_ratio=0.5, high_pass=False, order=3.0, channel_axis=None, squared_butterworth=True, npad=0)
            Filtered_image = skimage.filters.butterworth(Filtered_image, cutoff_frequency_ratio=0.5, high_pass=True, order=3.0, channel_axis=None, squared_butterworth=True, npad=0)
            erosion_dst = cv2.erode(Filtered_image, kernel_erode)

            # Contours
            erosion_dst = erosion_dst.astype(np.uint8)
            contours, _  = cv2.findContours(erosion_dst, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            return contours,img_gray_smooth_treshed
        
        case 'fourier':
            # filtering inn fourier's space
            img_gray_fourrierfiltered = FourrierFiltering (img_gray)

            # Thresholding image
            _,thresh   = cv2.threshold(img_gray_fourrierfiltered,Tresh[0],Tresh[1],cv2.THRESH_BINARY)

            # def kernel for erosion / dilatation
            kernel_dilate = np.ones((kernel_size,kernel_size),np.uint8)
            kernel_erode  = np.ones((kernel_size+2,kernel_size+2),np.uint8)
            
            #erosion dilatation
            dilate      = cv2.dilate(thresh,kernel_dilate,iterations = 1)
            erosion_dst = cv2.erode(dilate,kernel_erode,iterations = 1)

            # Contours
            erosion_dst = erosion_dst.astype(np.uint8)
            contours, _  = cv2.findContours(erosion_dst, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            return contours,thresh
        
        case 'bitwise_fourier_AND_Treshhold':
            # filtering inn fourier's space
            _,thresh0   = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_gray_fourrierfiltered = FourrierFiltering (thresh0)

            # Thresholding image
            _,thresh1   = cv2.threshold(img_gray,Tresh[0],Tresh[1],cv2.THRESH_BINARY)
            thresh = cv2.bitwise_and(thresh0,thresh1)
            # def kernel for erosion / dilatation
            kernel_dilate = np.ones((kernel_size,kernel_size),np.uint8)
            kernel_erode  = np.ones((kernel_size+2,kernel_size+2),np.uint8)
            
            #erosion dilatation
            dilate      = cv2.dilate(thresh,kernel_dilate,iterations = 1)
            erosion_dst = cv2.erode(dilate,kernel_erode,iterations = 1)

            # Contours
            erosion_dst = erosion_dst.astype(np.uint8)
            contours, _  = cv2.findContours(erosion_dst, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            return contours,thresh
        
        case other:
            easygui.msgbox('The method for fincontours is unknown. Check value.', 'error')

def drawbox (contours):
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return x,y,h,w

def computeFES(w_of_box, h_of_box):
    FES = 4/math.pi*np.arctan(w_of_box/h_of_box)-1
    return FES

def calibration_processing (CalibrationValueFES, CalibrationValueZ, Calib_analysis):
    from scipy.signal import savgol_filter
    CalibrationValueFES_SavGolFil   = savgol_filter(CalibrationValueFES, 51, 3)
    min_index = np.argmin(CalibrationValueFES_SavGolFil)
    max_index = np.argmax(CalibrationValueFES_SavGolFil)
    residue = CalibrationValueFES-CalibrationValueFES_SavGolFil

    if min_index < max_index:
        min_index_corr = min_index
        max_index_corr = max_index
    elif min_index > max_index:
        min_index_corr = max_index
        max_index_corr = min_index
    else:
        print ("probleme during calibration processing!")
        
    diffs = np.diff(CalibrationValueFES_SavGolFil[min_index_corr:max_index_corr])
    if np.count_nonzero(np.diff(np.sign(diffs))) < 2:
        monotonousFES = 1
        monotonic_FES = CalibrationValueFES_SavGolFil
        print ('the FES function is monotone')
    else:
        monotonousFES = 0
        from scipy.interpolate import PchipInterpolator
        monotonic_FES = CalibrationValueFES_SavGolFil
        # monotonic_FES[min_index_corr:max_index_corr] = PchipInterpolator(range (min_index_corr,max_index_corr),CalibrationValueFES_SavGolFil[min_index_corr:max_index_corr])
        print ('the FES function is not monotone')

    precision_FES   = np.max(residue[min_index_corr:max_index_corr])-np.min(residue[min_index_corr:max_index_corr])
    facteurEchelle  = (CalibrationValueZ[max_index_corr-25]-CalibrationValueZ[min_index_corr+25])/(CalibrationValueFES_SavGolFil[max_index_corr-25]-CalibrationValueFES_SavGolFil[min_index_corr+25])
    precision_z     = facteurEchelle * precision_FES
    IntervalOfFES   = (min_index_corr,max_index_corr)

    if Calib_analysis == True:
        import matplotlib.pyplot as plt
        plt.subplot(2,1,1)
        plt.plot (CalibrationValueZ, CalibrationValueFES,linewidth=4, label='Computed FES')
        plt.plot (CalibrationValueZ, CalibrationValueFES_SavGolFil,'--',linewidth=2, label='Savitzky-Golay filter estimate of FES')
        plt.ylabel ('FES',fontweight='bold',fontsize=14)
        plt.legend (loc="upper left",fontsize=10)

        plt.subplot(2,2,3)
        plt.plot (CalibrationValueZ, residue)
        plt.xlabel ('Defocus in µm',fontweight='bold',fontsize=14)
        plt.ylabel ('residue',fontweight='bold',fontsize=14)
        plt.subplot(2,2,4)
        plt.boxplot(residue)

        plt.figure()
        plt.plot (CalibrationValueZ, CalibrationValueFES,linewidth=4, label='Computed FES')
        plt.plot (CalibrationValueZ, CalibrationValueFES_SavGolFil,'--',linewidth=2, label='Savitzky-Golay filter estimate of FES')
        plt.plot (CalibrationValueZ, monotonic_FES,'--',linewidth=2, label='Monotonic estimation pf FES')
        plt.ylabel ('FES',fontweight='bold',fontsize=14)
        plt.legend (loc="upper left",fontsize=10)
        print("the resolution error is = " + str(precision_z))

    return CalibrationValueFES_SavGolFil, precision_z, monotonousFES, IntervalOfFES

def FourrierFiltering (img_gray):

    fourier = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    # Shift the zero-frequency component to the center of the spectrum
    fourier_shift = np.fft.fftshift(fourier)
    row, col = img_gray.shape
    center_row, center_col = row // 2, col // 2

    # Creation de Masque 
    mask = np.zeros((row, col, 2), np.uint8)
    size_FFT_cross = col // 2
    size_FFT_cross2 = 15
    mask[center_row - size_FFT_cross2:center_row + size_FFT_cross2, (center_col) - size_FFT_cross:(center_col) + size_FFT_cross] = 1
    mask[center_row - size_FFT_cross2:center_row + size_FFT_cross2, (center_col) - size_FFT_cross:(center_col) + size_FFT_cross] = 1
    mask[(center_row) - size_FFT_cross:(center_row) + size_FFT_cross, center_col - size_FFT_cross2:center_col + size_FFT_cross2] = 1
    mask[(center_row) - size_FFT_cross:(center_row) + size_FFT_cross, center_col - size_FFT_cross2:center_col + size_FFT_cross2] = 1
    mask[center_row - size_FFT_cross2:center_row + size_FFT_cross2, center_col - size_FFT_cross2:center_col + size_FFT_cross2] = 0

    # calculate the magnitude of the Fourier Transform
    magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))

    # Scale the magnitude for display
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # calculate resulting image
    fft_shift = fourier_shift * mask
    fft_ifft_shift = np.fft.ifftshift(fft_shift)
    imageThen = cv2.idft(fft_ifft_shift)
    imageThen = cv2.magnitude(imageThen[:,:,0], imageThen[:,:,1])
    imageThen = cv2.normalize(src=imageThen, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return imageThen

def TestGranuImage(img_gray):
    contours,_ = cv2.findContours(img_gray,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Granu = len(contours)//3+2
    
    if Granu >= 25:
        kernel_size_expexted = 25
    else:
        kernel_size_expexted = Granu

    return Granu, kernel_size_expexted

def fixTreshvalue (img_gray) :
    Tresh_High =np.max(img_gray)
    if Tresh_High >= 255:
        Tresh_High = 255
    elif Tresh_High >= 200:
        Tresh_High  = np.max(img_gray)+1
    else:
        Tresh_High  = 200

    Tresh_Low   = math.floor(0.45 * Tresh_High)
    Tresh = (Tresh_Low,Tresh_High)
    return Tresh
# TBA
def Test_Img_sat (img_gray) :
    nb_de_pix_total = img_gray.shape[0]*img_gray.shape[1]
    nb_de_pix_sat = np.count_nonzero(img_gray >= 254)
    ratio = nb_de_pix_sat/nb_de_pix_total
    print('saturation à'+ str(ratio*100)[5]+'%')
    return ratio

def countblackPixel(Tresholded_Filtered_Img,x,y,h,w):
    white_pix = np.count_nonzero((Tresholded_Filtered_Img[y:y+h, x:x+w] == [255]))
    ratio = white_pix/(h*w)
    Decision_treshold = 0.6
    if ratio >= Decision_treshold:
        print("1 reflexion")
    else:
        print("more than 1 reflexion")
# GUI item; mostly pop-up for user imput :

# To DO class instead of functions
def MSGboxAF():
    # Define the button choices
    choices = ["FastAF", "Calibration", "live monitoring","StayPut"]

    # Show a button box and get the user's choice
    method = easygui.buttonbox("Choose a method:", choices=choices)

    # Check which button was clicked and take appropriate action
    if method:
        print(f"Method selected: {method}")
    else:
        print("No method selected.")
    return method

def ReadUserValue(msg):
    # Prompt the user to enter a value
    Target = easygui.enterbox("Enter a value:", title=msg)   
    return Target