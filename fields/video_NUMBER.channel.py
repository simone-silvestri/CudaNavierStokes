# state file generated using paraview version 5.8.1

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# trace generated using paraview version 5.8.1
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1293, 803]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [0.9999998651328497, 3.1415926376357675, 6.283187659457326]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [7.135340416450747, -7.719236726663626, 20.174898714002133]
renderView1.CameraFocalPoint = [-0.003702255181213865, 3.337604588264623, 6.991081684341452]
renderView1.CameraViewUp = [0.9223527678430617, 0.2865784139184633, -0.25911037093914446]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 7.0591461228086745
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XDMF Reader'
fieldNUMBERxmf = XDMFReader(FileNames=['/home/simone/Dropbox/cuda-solvers/fields/field.NUMBER.xmf'])
fieldNUMBERxmf.PointArrayStatus = ['u', 'v', 'w']
fieldNUMBERxmf.GridStatus = ['T0000000']

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from fieldNUMBERxmf
fieldNUMBERxmfDisplay = Show(fieldNUMBERxmf, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'w'
wLUT = GetColorTransferFunction('w')
wLUT.RGBPoints = [-0.004936348663258822, 0.0, 0.0, 0.34902, 0.03712337830216869, 0.039216, 0.062745, 0.380392, 0.0791831052675962, 0.062745, 0.117647, 0.411765, 0.1212428322330237, 0.090196, 0.184314, 0.45098, 0.1633025591984512, 0.12549, 0.262745, 0.501961, 0.20536228616387872, 0.160784, 0.337255, 0.541176, 0.24742201312930623, 0.2, 0.396078, 0.568627, 0.28948174009473376, 0.239216, 0.454902, 0.6, 0.33154146706016124, 0.286275, 0.521569, 0.65098, 0.3736011940255887, 0.337255, 0.592157, 0.701961, 0.41566092099101626, 0.388235, 0.654902, 0.74902, 0.4577206479564438, 0.466667, 0.737255, 0.819608, 0.4997803749218713, 0.572549, 0.819608, 0.878431, 0.5418401018872987, 0.654902, 0.866667, 0.909804, 0.5838998288527263, 0.752941, 0.917647, 0.941176, 0.6259595558181538, 0.823529, 0.956863, 0.968627, 0.6680192827835814, 0.988235, 0.960784, 0.901961, 0.6680192827835814, 0.941176, 0.984314, 0.988235, 0.6949375080414548, 0.988235, 0.945098, 0.85098, 0.7218557332993285, 0.980392, 0.898039, 0.784314, 0.7521387367144362, 0.968627, 0.835294, 0.698039, 0.7941984636798638, 0.94902, 0.733333, 0.588235, 0.8362581906452913, 0.929412, 0.65098, 0.509804, 0.8783179176107188, 0.909804, 0.564706, 0.435294, 0.9203776445761463, 0.878431, 0.458824, 0.352941, 0.9624373715415738, 0.839216, 0.388235, 0.286275, 1.0044970985070014, 0.760784, 0.294118, 0.211765, 1.046556825472429, 0.701961, 0.211765, 0.168627, 1.0886165524378564, 0.65098, 0.156863, 0.129412, 1.1306762794032839, 0.6, 0.094118, 0.094118, 1.1727360063687116, 0.54902, 0.066667, 0.098039, 1.214795733334139, 0.501961, 0.05098, 0.12549, 1.2568554602995665, 0.45098, 0.054902, 0.172549, 1.2989151872649942, 0.4, 0.054902, 0.192157, 1.3409749142304215, 0.34902, 0.070588, 0.211765]
wLUT.ColorSpace = 'Lab'
wLUT.NanColor = [0.25, 0.0, 0.0]
wLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
fieldNUMBERxmfDisplay.Representation = 'Surface'
fieldNUMBERxmfDisplay.ColorArrayName = ['POINTS', 'w']
fieldNUMBERxmfDisplay.LookupTable = wLUT
fieldNUMBERxmfDisplay.OSPRayScaleArray = 'u'
fieldNUMBERxmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
fieldNUMBERxmfDisplay.SelectOrientationVectors = 'None'
fieldNUMBERxmfDisplay.ScaleFactor = 1.250092508
fieldNUMBERxmfDisplay.SelectScaleArray = 'u'
fieldNUMBERxmfDisplay.GlyphType = 'Arrow'
fieldNUMBERxmfDisplay.GlyphTableIndexArray = 'u'
fieldNUMBERxmfDisplay.GaussianRadius = 0.0625046254
fieldNUMBERxmfDisplay.SetScaleArray = ['POINTS', 'u']
fieldNUMBERxmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
fieldNUMBERxmfDisplay.OpacityArray = ['POINTS', 'u']
fieldNUMBERxmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
fieldNUMBERxmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
fieldNUMBERxmfDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
fieldNUMBERxmfDisplay.OSPRayScaleFunction.Points = [-6.608525935006737, 0.0, 0.5, 0.0, 50.675240851637405, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
fieldNUMBERxmfDisplay.ScaleTransferFunction.Points = [-0.0599910441264529, 0.0, 0.5, 0.0, 0.05998621903040874, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
fieldNUMBERxmfDisplay.OpacityTransferFunction.Points = [-0.0599910441264529, 0.0, 0.5, 0.0, 0.05998621903040874, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for wLUT in view renderView1
wLUTColorBar = GetScalarBar(wLUT, renderView1)
wLUTColorBar.Title = 'w'
wLUTColorBar.ComponentTitle = ''

# set color bar visibility
wLUTColorBar.Visibility = 1

# show color legend
fieldNUMBERxmfDisplay.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'w'
wPWF = GetOpacityTransferFunction('w')
wPWF.Points = [-0.004936348663258822, 0.0, 0.5, 0.0, 1.3409749142304215, 1.0, 0.5, 0.0]
wPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# finally, restore active source
SetActiveSource(fieldNUMBERxmf)
# ----------------------------------------------------------------

WriteImage("image.NUMBER.png")

