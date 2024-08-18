from REACTRL import RealExpEnv, Episode_Memory, Createc_Controller, sac_agent, ReplayMemory, HerReplayMemory
# default scan values
    # NumX = 512
    # NumY = 512
    # DeltaX = 119
    # DeltaY = 119
    # DX/DDeltaX = 33
    # Channelselectval = 3
    # ScanXMode = 0
    # CHMode = 0
    # BiasVolt.[mV] = 100
    # FBLogIset = 50

basic_params=['Delta X [Dac]', 'Delta Y [Dac]', 'Num.X', 'Num.Y', 'Delay X+', 'Delay X-', 'Delay Y', 'DX/DDeltaX','GainX','GainY','GainZ', 'Rotation', 'BiasVolt.[mV]', 'GainPre 10^', 'ScanChannels', 'PlanDx', 'PlanDy', 'OffsetX', 'OffsetY', 'ZPiezoconst', 'XPiezoconst', 'YPiezoconst', 'RepeatRotinc', 'RptBVoltinc', 'Repeatinterval', 'Repeatcounter', 'RepeatXoffset', 'RepeatYoffset', 'Scantype', 'ScanXMode', 'Scancoarse', 'CHMode', 'Channels', 'Preamptype', 'VFBMode', 'RotCMode', 'ScanYMode', 'ScanYDirec']
def get_basic_params():
    params=pd.DataFrame(basic_params, columns=['basic params'])
    params.loc[:,'values']=[env.createc_controller.stm.getparam(i) for i in basic_params]
    return params

def set_scan_params():
    # Set X x Y image size (Pixels) #
    env.createc_controller.stm.setparam('Num.X', NumX)
    env.createc_controller.stm.setparam('Num.Y', NumY)

    # Set the X x Y image size (Å)
    env.createc_controller.stm.setparam('Delta X [DAC]', DeltaX)
    env.createc_controller.stm.setparam('Delta Y [DAC]', DeltaY)

    # Set the speed #
    env.createc_controller.stm.setparam('DX/DDeltaX', DX/DDeltaX)
    
    # Set the Topography and Current channels #
    env.createc_controller.stm.setparam('Channelselectval', Channelselectval)

    # Set to Forward + Backward
    env.createc_controller.stm.setparam('ScanXMode', ScanXMode)

    # Set to Constant Current mode #
    env.createc_controller.stm.setparam('CHMode', CHMode)

    # Set Bias Voltage (mV) #
    env.createc_controller.stm.setparam('BiasVolt.[mV]', BiasVolt.[mV])

    # Set Setpoint Current (pA) #
    env.createc_controller.stm.setparam('FBLogIset', FBLogIset)

def set_tipform_params():

    env.createc_controller.stm.setparam('TipForm_Volt', -3) # Set the bias voltage for the atom manipulation(unit V)
    env.createc_controller.stm.setparam('TipForm_Z', 0.0) # Set the z position for the atom manipulation(unit angstrom)
    env.createc_controller.stm.setparam('Tip_Delay', 0.4915) # Set the delay time for the atom manipulation(unit s)
    env.createc_controller.stm.setparam('Tip_Latddx', 833) # Set the lateral ddx for the atom manipulation(unit angstrom/ms^2)
    env.createc_controller.stm.setparam('Tip_LatDelay', 20) # Set the lateral delay for the atom manipulation(unit ms)
    env.createc_controller.stm.setparam('Tip_Gain', 6) # Set the gain for the atom manipulation(unit 1)
    env.createc_controller.stm.setparam('TipForm_Zoffset', 0.0) # Set the z offset for the atom manipulation(unit angstrom)

# tip_condition_actions={'stall': [0, 0],
#                       'soft pulse pos':[0, 2.5],
#                       'medium pulse pos': [0, 4.0],
#                       'soft pulse neg': [0 , -2.5], 
#                       'medium pulse neg': [0, -4],
#                       'strong pulse neg': [0, -8],
#                       'very soft dip': [0, -8],
#                       'soft dip 1': [-0.80, 0.02],
#                       'soft dip 2': [-1.2, 0.02],
#                       'soft dip 3': [-1.8, 0.02],
#                       'medium soft': [-2.5, 0.02],
#                       'strong dip': [-5.0, 0.02]
#                       }

def set_movement_params():
    env.createc_controller.ramp_bias_mV(mvoltage)
    preamp_grain = 10**float(env.createc_controller.stm.getparam("Latmangain"))
    env.createc_controller.stm.setparam("LatmanVolt",  mvoltage) #(mV)
    env.createc_controller.stm.setparam("Latmanlgi", pcurrent*1e-9*preamp_grain) #(pA)
    args = x_start_pixel,y_start_pixel,x_end_pixel,y_end_pixel
    env.createc_controller.stm.latmanip(*args) #unit: image pixel
    #Channel: 0: time in sec 1: X 2: Y 3: Current I 4: dI/dV 5: d2I/dV 6: ADC0 7: ADC1 8: ADC2 9: ADC3 10: df 11: Damping 12: Amplitude 13: di_q 14: di2_q 15: Topography(DAC0) 16: CP(DAC1)
    #Units: 0: Default 1: Volt 2: DAC 3: Ampere 4: nm 5: Hz


def set_diss_params():
    mvoltage = 50
    pcurrent = 500
    z_nm = 8
    env.createc_controller.stm.setparam('BiasVolt.[mV]',mvoltage)
    env.createc_controller.ramp_bias_mV(mvoltage)
    preamp_grain = 10**float(self.stm.getparam("Latmangain"))
    env.createc_controller.stm.setparam("LatmanVolt",  mvoltage) #(mV)
    env.createc_controller.stm.setparam("Latmanlgi", pcurrent*1e-9*preamp_grain) #(pA)

    env.createc_controller.set_Z_approach(z_nm)
    args = x_start_nm, y_start_nm, x_end_nm, y_end_nm, offset_nm, len_nm 
    x_start_pixel, y_start_pixel, _, _ = self.nm_to_pixel(*args)
    env.createc_controller.stm.btn_tipform(x_start_pixel, y_start_pixel) #unit: image pixel
    env.createc_controller.stm.waitms(50)

#def set_ass_params():

#def get_tipform_params():





#def get_scan_params():


### Load CreaTec scan parameters ###
# This assumes a 512x512 pixel size with 400x400 Å size and a scanning speed equivalent to about 5-6 minutes per image. Why?
# This is approximately the limit of reasonable CO imaging and gives you a large area to test with. Too large of an image
# may yield odd CO images. The program will attempt to scale smaller images as well, but your results may be poor.

