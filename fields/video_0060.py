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
renderView1.CenterOfRotation = [3.141592502593994, 3.141592502593994, 3.141592502593994]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-11.724451399466377, -8.531669812706127, -6.0639506121361375]
renderView1.CameraFocalPoint = [3.141592502593997, 3.1415925025939937, 3.141592502593998]
renderView1.CameraViewUp = [0.28783255702859856, 0.3395933495999064, -0.8954489242954589]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 5.441397831170258
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
field128xmf = XDMFReader(FileNames=['/home/simone/Dropbox/cuda-solvers/fields/field.0060.xmf'])
field128xmf.PointArrayStatus = ['u', 'v', 'w']
field128xmf.GridStatus = ['T0000000']

# create a new 'Calculator'
calculator1 = Calculator(Input=field128xmf)
calculator1.Function = 'u*iHat+v*jHat+w*kHat'

# create a new 'Gradient Of Unstructured DataSet'
gradientOfUnstructuredDataSet1 = GradientOfUnstructuredDataSet(Input=calculator1)
gradientOfUnstructuredDataSet1.ScalarArray = ['POINTS', 'Result']
gradientOfUnstructuredDataSet1.ComputeGradient = 0
gradientOfUnstructuredDataSet1.ComputeVorticity = 1

# create a new 'Calculator'
calculator2 = Calculator(Input=gradientOfUnstructuredDataSet1)
calculator2.ResultArrayName = 'vortZ'
calculator2.Function = 'Vorticity_Z'

# create a new 'Resample To Image'
resampleToImage1 = ResampleToImage(Input=calculator2)
resampleToImage1.SamplingDimensions = [192, 128, 160]
resampleToImage1.SamplingBounds = [0.0, 3.1415925, 0.0, 6.283185, 0.0, 5.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from calculator1
calculator1Display = Show(calculator1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
calculator1Display.Representation = 'Outline'
calculator1Display.ColorArrayName = ['POINTS', '']
calculator1Display.OSPRayScaleArray = 'r'
calculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator1Display.SelectOrientationVectors = 'Result'
calculator1Display.ScaleFactor = 0.6283185
calculator1Display.SelectScaleArray = 'r'
calculator1Display.GlyphType = 'Arrow'
calculator1Display.GlyphTableIndexArray = 'r'
calculator1Display.GaussianRadius = 0.031415925
calculator1Display.SetScaleArray = ['POINTS', 'r']
calculator1Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator1Display.OpacityArray = ['POINTS', 'r']
calculator1Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator1Display.DataAxesGrid = 'GridAxesRepresentation'
calculator1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
calculator1Display.OSPRayScaleFunction.Points = [-6.608525935006737, 0.0, 0.5, 0.0, 50.675240851637405, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator1Display.ScaleTransferFunction.Points = [0.9947584288675436, 0.0, 0.5, 0.0, 1.0052415711324563, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator1Display.OpacityTransferFunction.Points = [0.9947584288675436, 0.0, 0.5, 0.0, 1.0052415711324563, 1.0, 0.5, 0.0]

# show data from resampleToImage1
resampleToImage1Display = Show(resampleToImage1, renderView1, 'UniformGridRepresentation')

# get color transfer function/color map for 'vortZ'
vortZLUT = GetColorTransferFunction('vortZ')
vortZLUT.RGBPoints = [-1.9818024619940908, 0.0862745098039216, 0.00392156862745098, 0.298039215686275, -1.8615695765483926, 0.113725, 0.0235294, 0.45098, -1.7617153925547913, 0.105882, 0.0509804, 0.509804, -1.6924286787641636, 0.0392157, 0.0392157, 0.560784, -1.6251797345276575, 0.0313725, 0.0980392, 0.6, -1.5599687915680207, 0.0431373, 0.164706, 0.639216, -1.4662281479595562, 0.054902, 0.243137, 0.678431, -1.3419195707026963, 0.054902, 0.317647, 0.709804, -1.1890814771964606, 0.0509804, 0.396078, 0.741176, -1.0899913540967554, 0.0392157, 0.466667, 0.768627, -0.9909012309970501, 0.0313725, 0.537255, 0.788235, -0.8874805078166363, 0.0313725, 0.615686, 0.811765, -0.7815127834379103, 0.0235294, 0.709804, 0.831373, -0.6755448273364488, 0.0509804, 0.8, 0.85098, -0.587917724113121, 0.0705882, 0.854902, 0.870588, -0.5064039295522198, 0.262745, 0.901961, 0.862745, -0.4350794462074459, 0.423529, 0.941176, 0.87451, -0.32503595099770455, 0.572549, 0.964706, 0.835294, -0.2516734663760738, 0.658824, 0.980392, 0.843137, -0.19868960418671056, 0.764706, 0.980392, 0.866667, -0.14163008702769986, 0.827451, 0.980392, 0.886275, -0.029548706402469094, 0.913725, 0.988235, 0.937255, 0.0050946504928506275, 1.0, 1.0, 0.972549019607843, 0.039738007388171015, 0.988235, 0.980392, 0.870588, 0.09068410002338756, 0.992156862745098, 0.972549019607843, 0.803921568627451, 0.12940311188835518, 0.992157, 0.964706, 0.713725, 0.19461405484799266, 0.988235, 0.956863, 0.643137, 0.29854400967258443, 0.980392, 0.917647, 0.509804, 0.3861713446186352, 0.968627, 0.87451, 0.407843, 0.47583644911881984, 0.94902, 0.823529, 0.321569, 0.5410473920784538, 0.929412, 0.776471, 0.278431, 0.6368260369637746, 0.909804, 0.717647, 0.235294, 0.722415370632963, 0.890196, 0.658824, 0.196078, 0.7927209847976362, 0.878431, 0.619608, 0.168627, 0.8918111078973403, 0.870588, 0.54902, 0.156863, 0.9909012309970453, 0.85098, 0.47451, 0.145098, 1.0899913540967494, 0.831373, 0.411765, 0.133333, 1.1890814771964549, 0.811765, 0.345098, 0.113725, 1.2881716002961594, 0.788235, 0.266667, 0.0941176, 1.387261723395864, 0.741176, 0.184314, 0.0745098, 1.4863518464955685, 0.690196, 0.12549, 0.0627451, 1.5854419695952726, 0.619608, 0.0627451, 0.0431373, 1.6781638399333139, 0.54902, 0.027451, 0.0705882, 1.7596775657015409, 0.470588, 0.0156863, 0.0901961, 1.8513805013072118, 0.4, 0.00392157, 0.101961, 1.9818024619940913, 0.188235294117647, 0.0, 0.0705882352941176]
vortZLUT.ColorSpace = 'Lab'
vortZLUT.NanColor = [0.25, 0.0, 0.0]
vortZLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'vortZ'
vortZPWF = GetOpacityTransferFunction('vortZ')
vortZPWF.Points = [-1.9818024619940908, 1.0, 0.5, 0.0, -1.575891137123108, 0.9625000357627869, 0.5, 0.0, -0.8834540843963623, 0.71875, 0.5, 0.0, -0.4775427579879761, 0.518750011920929, 0.5, 0.0, -0.2865256369113922, 0.09375, 0.5, 0.0, -0.09550853818655014, 0.0, 0.5, 0.0, 0.3104028105735779, 0.0062500000931322575, 0.5, 0.0, 0.5611127614974976, 0.11250000447034836, 0.5, 0.0, 1.026716947555542, 0.6500000357627869, 0.5, 0.0, 1.9818024619940913, 1.0, 0.5, 0.0]
vortZPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
resampleToImage1Display.Representation = 'Volume'
resampleToImage1Display.ColorArrayName = ['POINTS', 'vortZ']
resampleToImage1Display.LookupTable = vortZLUT
resampleToImage1Display.OSPRayScaleArray = 'vortZ'
resampleToImage1Display.OSPRayScaleFunction = 'PiecewiseFunction'
resampleToImage1Display.SelectOrientationVectors = 'Result'
resampleToImage1Display.ScaleFactor = 0.6283185
resampleToImage1Display.SelectScaleArray = 'vortZ'
resampleToImage1Display.GlyphType = 'Arrow'
resampleToImage1Display.GlyphTableIndexArray = 'vortZ'
resampleToImage1Display.GaussianRadius = 0.031415925
resampleToImage1Display.SetScaleArray = ['POINTS', 'vortZ']
resampleToImage1Display.ScaleTransferFunction = 'PiecewiseFunction'
resampleToImage1Display.OpacityArray = ['POINTS', 'vortZ']
resampleToImage1Display.OpacityTransferFunction = 'PiecewiseFunction'
resampleToImage1Display.DataAxesGrid = 'GridAxesRepresentation'
resampleToImage1Display.PolarAxes = 'PolarAxesRepresentation'
resampleToImage1Display.ScalarOpacityUnitDistance = 0.08569130435712326
resampleToImage1Display.ScalarOpacityFunction = vortZPWF
resampleToImage1Display.IsosurfaceValues = [1.1102230246251565e-16]
resampleToImage1Display.SliceFunction = 'Plane'
resampleToImage1Display.Slice = 63

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
resampleToImage1Display.OSPRayScaleFunction.Points = [-6.608525935006737, 0.0, 0.5, 0.0, 50.675240851637405, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
resampleToImage1Display.ScaleTransferFunction.Points = [-1.9818024619940908, 0.0, 0.5, 0.0, 1.9818024619940913, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
resampleToImage1Display.OpacityTransferFunction.Points = [-1.9818024619940908, 0.0, 0.5, 0.0, 1.9818024619940913, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
resampleToImage1Display.SliceFunction.Origin = [3.1415925, 3.1415925, 3.1415925]

# setup the color legend parameters for each legend in this view

# get color legend/bar for vortZLUT in view renderView1
vortZLUTColorBar = GetScalarBar(vortZLUT, renderView1)
vortZLUTColorBar.Title = 'vortZ'
vortZLUTColorBar.ComponentTitle = ''

# set color bar visibility
vortZLUTColorBar.Visibility = 1

# show color legend
resampleToImage1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# finally, restore active source
SetActiveSource(calculator1)
# ----------------------------------------------------------------


WriteImage("image.0060.png")
