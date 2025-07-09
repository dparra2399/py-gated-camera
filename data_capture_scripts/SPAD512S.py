"""

@author: Pi Imaging Technology

This file contains basic functions to interact with a SPAD512S system connected to
its software. Creating multiple instances of the class allows to control several
SPAD512S systems independently.

"""

import socket
import errno
import numpy as np
import time



class SPAD512S:
    def __init__(self, port):
        '''
        This initializes the SPAD512S class and establishes the TCPIP connection.

        Parameters
        ----------
        port : int
            TCPIP port to be connected to.

        Raises
        ------
        serr
            Checks if port is valid and connection can be established.

        '''
        self.t = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.t.connect(('127.0.0.1', port))
        except socket.error as serr:
            if serr.errno != errno.ECONNREFUSED:
                raise serr
            print('error!!!!!!!')
            return None
        time.sleep(1)
        msg = self.t.recv(8192).decode('utf-8')
        self.detType = msg.split()[0]
        sendInfo = bytes("D", " utf8 ")
        self.t.send(sendInfo)
        self.info = self.t.recv(8191).decode('utf-8').split('\n')
        self.row = 512
        self.col = [4, 8, 16, 32, 64 , 128, 256, 512]
        self.nbrPix = self.row*self.col
        self.intBitDepths = [1,4,6,7,8,9,10,11,12]
        self.gatedBitDepths = [6,7,8, 9, 10, 11, 12]


    '''
                ------------ Setting up functions ------------
    ''' 


    def set_Vex(self, Vex):
        '''
        This function sets the excess bias of the SPAD512S system.
        
        Parameters
        ---- ------
        Vex : float
            Excess bias of the SPAD512S system

        Returns
        -------
        msg : string
            Voltage changed confirmation message

        '''
        commandV = bytes("V," + str(Vex), "utf-8")
        self.t.send(commandV)
        msg = self.t.recv(8191).decode('utf-8')
        return msg
    
    def get_voltages(self):
        '''
        Requests the operating voltages of the SPAD512S system.
        

        Returns
        -------
        Vq : string
            Current quenching voltage.
        Vex : string
            Current excess voltage.

        '''
        commandV = bytes("V", "utf-8")
        self.t.send(commandV)
        msg = self.t.recv(8191).decode('utf-8').split(',')
        Vq = msg[0]
        Vex = msg[1]
        return Vq, Vex


    def get_info(self):
        '''
        Requests information about the connected SPAD512S system.

        Returns
        -------
        info : string list
            Returns: 
            - Master FPGA serial
            - Slave FPGA serial
            - Software version
            - Firmware version
            - Hardware version
            - Hardware flavour
            - Intensity imaging enabled
            - Gated imaging enabled
            - FLIM imaging enabled
        '''
        return self.info
    
    def get_temps(self):
        '''
        Requests SPAD512S system operating temperatures

        Returns
        -------
        T_MSTR : string 
            Current master FPGA temperature
        T_SLV : string
            Current slave FPGA temperature
        T_PCB : string
            Current PCB temperature
        T_CHIP: string
            Current chip temperature
        '''
        commandR = bytes("R", " utf8 ")
        self.t.send(commandR)
        temperatures = self.t.recv(8191).decode('utf-8').split(',')[0:4]
        T_MSTR = temperatures[0]
        T_SLV = temperatures[1]
        T_PCB = temperatures[2]
        T_CHIP = temperatures[3]
        return T_MSTR, T_SLV, T_PCB, T_CHIP
    

    def get_freq(self):
        '''
        Requests SPAD512S system triggers frequencies
        
        Returns
        -------
        freq : string list
            Returns: 
            - Laser clock frequency
            - Frame clock frequency
        '''
        commandR = bytes("R", " utf8 ")
        self.t.send(commandR)
        freq = self.t.recv(8191).decode('utf-8').split(',')[4:]
        return freq
    
    
    def set_exposure_mode(self, mode, intTime):
        '''
        Sets exposure mode to manual or automatic.

        Parameters
        ----------
        mode : bool
            0: manual exposure
            1: automatic exposure
        intTime : float
            Integration time of the liveview in manual exposure mode.

        Returns
        -------
        msg : string
            Confirmation message.

        '''
        commandAE = bytes("AE," + str(mode) + "," + str(intTime), "utf-8")
        self.t.send(commandAE)
        msg = self.t.recv(8191).decode('utf-8')
        return msg
    
    def calib_noise(self):
        '''
        Executes hot pixel calibration. Needs to be called when sensor is in
        the dark.

        Returns
        -------
        msg : string
            Calibration completed message.

        '''
        commandCalNoise = bytes("CALIB," + str(0), "utf-8")
        self.t.send(commandCalNoise)
        msg = self.t.recv(8191).decode('utf-8')
        return msg
    
    def calib_dead(self):
        '''
        Executes dead pixel calibration. Needs to be called when sensor is in
        the dark.

        Returns
        -------
        msg : string
            Calibration completed message.

        '''
        commandCalDead = bytes("CALIB," + str(1), "utf-8")
        self.t.send(commandCalDead)
        msg = self.t.recv(8191).decode('utf-8')
        return msg
    
    def calib_mst_slv_off(self):
        '''
        Executes master/slave offset calibration. Needs to be called when sensor
        is uniformely illuminated with a pulsed source.

        Returns
        -------
        msg : string
            Calibration completed message.

        '''
        commandCalOff = bytes("CALIB," + str(2), "utf-8")
        self.t.send(commandCalOff)
        msg = self.t.recv(8191).decode('utf-8')
        return msg
    
    def calib_breakdown(self):
        '''
        Executes breakdown calibration.

        Returns
        -------
        msg : string
            Calibrated breakdown value.

        '''
        commandCalBD = bytes("CALIB," + str(3), "utf-8")
        self.t.send(commandCalBD)
        msg = self.t.recv(8191).decode('utf-8')
        time.sleep(15)
        msg = self.t.recv(8191).decode('utf-8')
        return msg
    
    def enable_cooling(self, enable):
        '''
        Enables/disables SPAD512S cooling system.
        
        Parameters
        ----------
        enable : bool
            0 to disable cooling
            1 to enable cooling

        Returns
        -------
        msg : string
            Cooling disabling/enabling message.

        '''
        commandCooling = bytes("S," + str(enable), "utf-8")
        self.t.send(commandCooling)
        msg = self.t.recv(8191).decode('utf-8')
        return msg
    
    def set_path(self, path):
        '''
        This function sets the data saving directory.
        
        Parameters
        ---- ------
        path : string
            Path to store data in

        Returns
        -------
        msg : string
            Command returns the folder path if it exists

        '''
        commandPath = bytes("D," + str(path), "utf-8")
        self.t.send(commandPath)
        msg = self.t.recv(8191).decode('utf-8')
        return msg
    
    '''
          ------------ Intensity mode related functions ------------
    ''' 
    
    def get_intensity(self, iterations, intTime, bitDepth, overlap, timeout, pileup, im_width):
        '''
        Starts an intensity measurement with the SPAD512S system, and returns true shifted values.

        Parameters
        ----------
        iterations : int
            Defines the number of measurements to repeat and average.
        intTime : float
            Defines the integration time for the measurement
            (in ms for 6, 7, 8bits and in us for 1, 4bits).
        bitDepth : int
            Defines the bit depth of the generated images.
            Valid values are 1,4,6,7,8,9,10,11,12.
        overlap : int
            1 for enabling read/exposure overlap, 0 otherwise
        timeout : int
            1 for enabling timeout in the function to redo measurement after measurement failed, 0 otherwise
        pileup : int
            1 for pileup corrected values, 0 otherwise
        im_width : int
            Valid values are 4, 8, 16, 32, 64 , 128, 256, 512
            
        
        Returns
        -------
        img : bool array if bitDepth == 1, array of uint8 or uint16 depending on the bitDepth and pileup mode
            Arrays (size 512x512xiterations) of  photon counts in each 
            individual SPAD pixel.

        '''
        
        
        command = bytes("PU," + str(pileup), "utf8")
        self.t.send(command)
        msg = self.t.recv(32768)
        
        if not (bitDepth in self.intBitDepths): 
            print("Chosen bit depth is invalid. Please use one of the following"
                  " values: %s.\nDefault value of 8bit is used instead!"
                  % self.intBitDepths)
            bitDepth = 8
        if not (im_width in self.col): 
            print("Chosen image width is invalid. Please use one of the following"
                  " values: %s.\nDefault width of 512 is used instead!"
                  % self.col)
            im_width = 512
        command = bytes("I," + str(bitDepth) + ","+ str(intTime) + ","  +
                        str(iterations) + "," + "0," + str(overlap) + ",0" + ",1", "utf8")
        self.t.send(command)
        if timeout:
            self.t.settimeout(5.0)  # Set a timeout of 5 seconds
        else:
            self.t.settimeout(None)  # Remove the timeout
    
        time.sleep(1)
        data = bytearray()
        
        # Define data type
        if bitDepth == 1:
            img = np.empty([self.row,im_width,iterations], dtype = np.bool_)
        elif bitDepth == 4 or (bitDepth < 9 and pileup == 0):
            img = np.empty([self.row,im_width,iterations], dtype = np.uint8)
        else:
            img = np.empty([self.row,im_width,iterations], dtype = np.uint16)
        if bitDepth == 1:
            while 1:
                datablock = self.t.recv(32768) # try different buffer sizes
                data.extend(datablock)

                # we stream binary pixel values, either 0 for no data, or 255 for a 1 (each byte represents one pixel)

                if datablock[-4:] == bytearray("DONE", 'utf8'):

                    print("Process complete")
                    break

                elif datablock[-5:] == bytearray("ERROR", 'utf8'):

                    print(datablock[-160:])
                    print("Completed the run with errors")


            # remove DONE from the end of the data
            data = data[:-4]
            for i in range(iterations): 
                img_index_old = i*512*64
                img_index = (i+1)*512*64

                dataint = np.array(data[img_index_old:img_index], dtype = "uint8")
                databit = np.unpackbits(dataint)
                databit = np.rot90(databit.reshape((512, 512)))
                img[:,:,i] = databit
        else:
            if pileup:
                while 1:
                    datablock = self.t.recv(32768) 
                    data.extend(datablock)
                    if data[-4:] == bytearray("DONE", 'utf8'):
                        break
                data = data[:-4]
                img_index = 0
                for i in range(iterations):
                    img_index_old = img_index
                    img_index = (i+1)*self.row*im_width*2
                    if img_index - img_index_old > 10:
                        np_data = np.asarray(data[img_index_old:img_index])
                        array_262144_even = np_data[::2].astype(np.uint8)
                        array_262144_odd = np_data[1::2].astype(np.uint8)
                        new_array = (array_262144_odd * (2**8)) + (array_262144_even)
                        img[:,:,i] = new_array.reshape((512, im_width))                    
                    
            else :
                while 1:
                    datablock = self.t.recv(32768)
                    data.extend(datablock)
                    if data[-4:] == bytearray("DONE", 'utf8'):
                        break
                data = data[:-4]
                img_index = 0
                if bitDepth < 9:
                    for i in range(iterations):
                        img_index_old = img_index
                        img_index = (i+1)*self.row*im_width
                        if img_index - img_index_old > 10:
                            np_data = np.asarray(data[img_index_old:img_index])
                            img[:,:,i] = np.reshape(np_data,[self.row,im_width])
                else:
                    for i in range(iterations):
                        img_index_old = img_index
                        img_index = (i+1)*self.row*im_width*2
                        if img_index - img_index_old > 10:
                            np_data = np.asarray(data[img_index_old:img_index])
                            array_262144_even = np_data[::2].astype(np.uint8)
                            array_262144_odd = np_data[1::2].astype(np.uint8)
                            new_array = (array_262144_odd * (2**8)) + (array_262144_even)
                            img[:,:,i] = new_array.reshape((512, im_width))


        return img  
    
    '''
             ------------ Gated mode related functions ------------
    ''' 
    def get_gated_intensity(self, bitDepth, intTime, iterations, gate_steps, gate_step_size, gate_step_arbitrary, gate_width, 
                                    gate_offset, gate_direction, gate_trig, overlap, stream, pileup, im_width):
        '''
        Starts an intensity measurement with the SPAD512S system, and returns true shifted values.

        Parameters
        ----------
        iterations : int
            Defines the number of measurements to repeat and average.
        intTime : float
            Defines the integration time for the measurement
            (in ms for 6, 7, 8bits and in us for 1, 4bits).
        bitDepth : int
            Defines the bit depth of the generated images.
            Valid values are 1,4,6,7,8,9,10,11,12.
        im_width : int
            Valid values are 4, 8, 16, 32, 64 , 128, 256, 512

        Returns
        -------
        img : array of uint8 or uint16 depending on the bitDepth and pileup mode
            Arrays (size 512x512xiterations) of  photon counts in each 
            individual SPAD pixel.

        '''
        
        command = bytes("PU," + str(pileup), "utf8")
        self.t.send(command)
        print('hello')
        msg = self.t.recv(32768)
        print('world')
        if not (bitDepth in self.intBitDepths and bitDepth != 1 and bitDepth != 4 ): 
            print("Chosen bit depth is invalid. Please use one of the following"
                  " values: %s.\nDefault value of 8bit is used instead!"
                  % self.intBitDepths)
            bitDepth = 8
        if not (im_width in self.col): 
            print("Chosen image width is invalid. Please use one of the following"
                  " values: %s.\nDefault width of 512 is used instead!"
                  % self.col)
            im_width = 512
        command = bytes("G," + str(bitDepth) + "," + str(intTime) + "," + 
                        str(iterations) + ","+ str(gate_steps) + ","  + 
                        str(gate_step_size)+ "," + str(gate_step_arbitrary) +
                        ',' + str(gate_width) + "," + str(gate_offset) + ","+
                        str(gate_direction) + ","  + str(gate_trig)+ "," +
                        str(overlap) + ","+ str(stream),  " utf8 ")
        self.t.send(command)
        data = bytearray()
        # Define data type
        if bitDepth == 4 or (bitDepth < 9 and pileup == 0):
            img = np.empty([self.row,im_width,iterations*gate_steps], dtype = np.uint8)
        else:
            img = np.empty([self.row,im_width,iterations*gate_steps], dtype = np.uint16)
    
        if pileup:
            while 1:
                datablock = self.t.recv(32768) # error
                data.extend(datablock)
                if data[-4:] == bytearray("DONE", 'utf8'):
                    break
            data = data[:-4]
            img_index = 0
            for i in range(iterations*gate_steps):
                img_index_old = img_index
                img_index = (i+1)*self.row*im_width*2
                if img_index - img_index_old > 10:
                    np_data = np.asarray(data[img_index_old:img_index])
                    array_262144_even = np_data[::2].astype(np.uint8)
                    array_262144_odd = np_data[1::2].astype(np.uint8)
                    # new_array = (array_262144_odd.astype(np.uint32) * (2 ** (bitDepth - 8))) + (array_262144_even.astype(np.uint32) >> (16 - bitDepth))
                    new_array = (array_262144_odd * (2**8)) + (array_262144_even)
                    img[:,:,i] = new_array.reshape((512, 512))                    
                
        else :
            while 1:
                datablock = self.t.recv(32768)
                data.extend(datablock)
                if data[-4:] == bytearray("DONE", 'utf8'):
                    print('data')
                    break
            data = data[:-4]
            img_index = 0
            if bitDepth < 9:
                print('got here')
                for i in range(iterations*gate_steps):
                    img_index_old = img_index
                    img_index = (i+1)*self.row*im_width
                    if img_index - img_index_old > 10:
                        np_data = np.asarray(data[img_index_old:img_index])
                        img[:,:,i] = np.reshape(np_data,[self.row,im_width])
                        # img[:,:,i] = img[:,:,i].astype(int) >> (8-bitDepth)
            else:
                for i in range(iterations*gate_steps):
                    img_index_old = img_index
                    img_index = (i+1)*self.row*im_width*2
                    if img_index - img_index_old > 10:
                        np_data = np.asarray(data[img_index_old:img_index])
                        array_262144_even = np_data[::2].astype(np.uint8)
                        array_262144_odd = np_data[1::2].astype(np.uint8)
                        # new_array = (array_262144_odd.astype(np.uint32) * (2 ** (bitDepth - 8))) + (array_262144_even.astype(np.uint32) >> (16 - bitDepth))
                        new_array = (array_262144_odd * (2**8)) + (array_262144_even)
                        img[:,:,i] = new_array.reshape((512, 512))
        return img
    
    def set_arbitrary_steps(self, steps):
        '''
        Defines an array of arbitrary gate step sizes to use in gated mode.

        Parameters
        ----------
        steps : float array
            Array of floats containing the arbitrary gate steps size to use.

        Returns
        -------
        msg : string
            Confirmation message.

        '''
        stepsString  = ';'.join(map(str, steps))
        commandGa = bytes("Ga," + stepsString, "utf-8")
        self.t.send(commandGa)
        msg = self.t.recv(8191).decode('utf-8')
        return msg
    
    
    def get_opt_gated_param(self, gateStepSize, gateWidth):
        '''
        Returns the optimal gated mode parameters to cover one gate cycle.

        Parameters
        ----------
        intTime : float
            Measurement 8bit integration time.
        stepSize : float
            Size of the gate steps. This defines the distance in time between
            each gate positions in 18ps increments.
        gateWidth : int
            Defines the gate width in time in ns.

        Returns
        -------
        nbrSteps : int
            Number of gate steps required to fully cover one gate cycle.
        offset : int
            Offset between laser trigger pulse and gate to cover one gate cycle.
        gateStepSize : float
            Minimum step size.

        '''
        commandGf = bytes("Gf," + str(1) + "," + str(gateStepSize) + ","
                          + str(gateWidth) + "," + str(1), " utf8 ")
        self.t.send(commandGf)
        msg = self.t.recv(8191).decode('utf8')
        msg = msg.splitlines()
        nbrSteps = msg[2]
        nbrSteps = nbrSteps.replace('The number of gate steps is ', '')
        nbrSteps = int(float(nbrSteps.replace(' ', '')))
        offset = msg[3]
        offset = offset.replace('The gate offset is ', '')
        offset = offset.replace(' ps', '')
        offset = int(float(offset.replace(' ', '')))
        stepSize = msg[4]
        stepSize = stepSize.replace('The minimum gate step size is ', '')
        stepSize = stepSize.replace(' ps', '')
        stepSize = int(float(stepSize.replace(' ', '')))
        return nbrSteps, offset, stepSize
        

    '''
            ------------ FLIM related functions ------------
    ''' 
    def calib_FLIM(self, mode, intTime, expTau, gateWidth):
        '''
        Executes FLIM IRF calibration.

        Parameters
        ----------
        mode : int
            Fluorescence decay type:
                0 for mono-exponential decay
                1 for bi-exponential decay
        intTime : float
            Measurement integration time.
        expTau : float
            Expected fluorescence lifetime.
        gateWidth : int
            0 for short gate
            1 for medium gate
            2 for long gate

        Returns
        -------
        msg : string
            Calibration result.

        '''
        commandFC = bytes("F,c" + str(mode) + ',' + str(intTime)+ ',' + str(expTau)
                          + ',' + str(gateWidth), "utf-8")
        self.t.send(commandFC)
        msg = self.t.recv(8191).decode('utf-8')
        return msg
    
    def get_FLIM(self, intTime, subsample, rawData):
        '''
        Gets FLIM measurement.

        Parameters
        ----------
        intTime : int
            8bit integration time
        subsample : int
            Ratio of gate steps subsampling
        rawData : boolean
            0 for image output
            1 for raw data output

        Returns
        -------
        msg : string
            Command returns the paths to the image files and phasor plot or
            returns the raw data directly.

        '''
        commandF = bytes("F,i" + str(intTime) + ',' + str(subsample)+ ','
                         + str(rawData), "utf-8")
        self.t.send(commandF)
        msg = self.t.recv(8191).decode('utf-8')
        return msg


    '''
              ------------ Advanced functions ------------
    ''' 