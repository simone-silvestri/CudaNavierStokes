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
renderView1.CameraPosition = [11.476821594132291, 18.09324371934177, -6.9573101164021685]
renderView1.CameraFocalPoint = [0.3937989257301684, 3.2522908847985246, 5.888051144018317]
renderView1.CameraViewUp = [0.8611022502822093, -0.2703780284895443, 0.4305794192353166]
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
fieldtestxmf = XDMFReader(FileNames=['/home/simone/Dropbox/cuda-solvers/fields2/field.0425.xmf'])
fieldtestxmf.PointArrayStatus = ['u', 'v', 'w']
fieldtestxmf.GridStatus = ['T0000000']

# create a new 'Calculator'
calculator1 = Calculator(Input=fieldtestxmf)
calculator1.ResultArrayName = 'vec'
calculator1.Function = 'u*iHat+v*jHat+w*kHat'

# create a new 'Slice'
slice3 = Slice(Input=calculator1)
slice3.SliceType = 'Plane'
slice3.HyperTreeGridSlicer = 'Plane'
slice3.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice3.SliceType.Origin = [0.9999998705, 3.1415927299999997, 12.5]
slice3.SliceType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice3.HyperTreeGridSlicer.Origin = [0.9999998705, 3.1415927299999997, 6.28318746]

# create a new 'Slice'
slice1 = Slice(Input=calculator1)
slice1.SliceType = 'Plane'
slice1.HyperTreeGridSlicer = 'Plane'
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [0.04, 3.1415927299999997, 6.28318746]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [0.9999998705, 3.1415927299999997, 6.28318746]

# create a new 'Slice'
slice2 = Slice(Input=calculator1)
slice2.SliceType = 'Plane'
slice2.HyperTreeGridSlicer = 'Plane'
slice2.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice2.SliceType.Origin = [0.9999998705, 0.03, 6.28318746]
slice2.SliceType.Normal = [0.0, 1.0, 0.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice2.HyperTreeGridSlicer.Origin = [0.9999998705, 3.1415927299999997, 6.28318746]

# create a new 'Gradient Of Unstructured DataSet'
gradientOfUnstructuredDataSet1 = GradientOfUnstructuredDataSet(Input=calculator1)
gradientOfUnstructuredDataSet1.ScalarArray = ['POINTS', 'vec']
gradientOfUnstructuredDataSet1.ComputeGradient = 0
gradientOfUnstructuredDataSet1.ComputeQCriterion = 1

# create a new 'Contour'
contour1 = Contour(Input=gradientOfUnstructuredDataSet1)
contour1.ContourBy = ['POINTS', 'Q-criterion']
contour1.Isosurfaces = [-2.0]
contour1.PointMergeMethod = 'Uniform Binning'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from slice1
slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'v'
vLUT = GetColorTransferFunction('v')
vLUT.AutomaticRescaleRangeMode = 'Never'
vLUT.RGBPoints = [-0.15, 0.0, 0.0, 0.34902, -0.140625, 0.039216, 0.062745, 0.380392, -0.13125, 0.062745, 0.117647, 0.411765, -0.12187499999999998, 0.090196, 0.184314, 0.45098, -0.11249999999999999, 0.12549, 0.262745, 0.501961, -0.103125, 0.160784, 0.337255, 0.541176, -0.09374999999999999, 0.2, 0.396078, 0.568627, -0.08437499999999999, 0.239216, 0.454902, 0.6, -0.075, 0.286275, 0.521569, 0.65098, -0.065625, 0.337255, 0.592157, 0.701961, -0.056249999999999994, 0.388235, 0.654902, 0.74902, -0.046875, 0.466667, 0.737255, 0.819608, -0.03749999999999998, 0.572549, 0.819608, 0.878431, -0.028124999999999997, 0.654902, 0.866667, 0.909804, -0.01874999999999999, 0.752941, 0.917647, 0.941176, -0.009374999999999994, 0.823529, 0.956863, 0.968627, 0.0, 0.941176, 0.984314, 0.988235, 0.0, 0.988235, 0.960784, 0.901961, 0.006000000000000005, 0.988235, 0.945098, 0.85098, 0.01200000000000001, 0.980392, 0.898039, 0.784314, 0.01874999999999999, 0.968627, 0.835294, 0.698039, 0.02812500000000001, 0.94902, 0.733333, 0.588235, 0.037500000000000006, 0.929412, 0.65098, 0.509804, 0.046875, 0.909804, 0.564706, 0.435294, 0.056249999999999994, 0.878431, 0.458824, 0.352941, 0.06562499999999999, 0.839216, 0.388235, 0.286275, 0.07500000000000004, 0.760784, 0.294118, 0.211765, 0.084375, 0.701961, 0.211765, 0.168627, 0.09375, 0.65098, 0.156863, 0.129412, 0.103125, 0.6, 0.094118, 0.094118, 0.11250000000000002, 0.54902, 0.066667, 0.098039, 0.12187499999999998, 0.501961, 0.05098, 0.12549, 0.13125, 0.45098, 0.054902, 0.172549, 0.14062499999999997, 0.4, 0.054902, 0.192157, 0.15, 0.34902, 0.070588, 0.211765]
vLUT.ColorSpace = 'Lab'
vLUT.NanColor = [0.25, 0.0, 0.0]
vLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
slice1Display.Representation = 'Surface'
slice1Display.ColorArrayName = ['POINTS', 'v']
slice1Display.LookupTable = vLUT
slice1Display.OSPRayScaleArray = 'u'
slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice1Display.SelectOrientationVectors = 'vec'
slice1Display.ScaleFactor = 1.2500925477594138
slice1Display.SelectScaleArray = 'u'
slice1Display.GlyphType = 'Arrow'
slice1Display.GlyphTableIndexArray = 'u'
slice1Display.GaussianRadius = 0.06250462738797069
slice1Display.SetScaleArray = ['POINTS', 'u']
slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
slice1Display.OpacityArray = ['POINTS', 'u']
slice1Display.OpacityTransferFunction = 'PiecewiseFunction'
slice1Display.DataAxesGrid = 'GridAxesRepresentation'
slice1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
slice1Display.OSPRayScaleFunction.Points = [-6.608525935006737, 0.0, 0.5, 0.0, 50.675240851637405, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice1Display.ScaleTransferFunction.Points = [-0.04220631921160562, 0.0, 0.5, 0.0, 0.04245531540693075, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice1Display.OpacityTransferFunction.Points = [-0.04220631921160562, 0.0, 0.5, 0.0, 0.04245531540693075, 1.0, 0.5, 0.0]

# show data from slice2
slice2Display = Show(slice2, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
slice2Display.Representation = 'Surface'
slice2Display.ColorArrayName = ['POINTS', 'v']
slice2Display.LookupTable = vLUT
slice2Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice2Display.SelectOrientationVectors = 'None'
slice2Display.ScaleFactor = -2.0000000000000002e+298
slice2Display.SelectScaleArray = 'None'
slice2Display.GlyphType = 'Arrow'
slice2Display.GlyphTableIndexArray = 'None'
slice2Display.GaussianRadius = -1e+297
slice2Display.SetScaleArray = [None, '']
slice2Display.ScaleTransferFunction = 'PiecewiseFunction'
slice2Display.OpacityArray = [None, '']
slice2Display.OpacityTransferFunction = 'PiecewiseFunction'
slice2Display.DataAxesGrid = 'GridAxesRepresentation'
slice2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
slice2Display.OSPRayScaleFunction.Points = [-6.608525935006737, 0.0, 0.5, 0.0, 50.675240851637405, 1.0, 0.5, 0.0]

# show data from slice3
slice3Display = Show(slice3, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
slice3Display.Representation = 'Surface'
slice3Display.ColorArrayName = ['POINTS', 'v']
slice3Display.LookupTable = vLUT
slice3Display.OSPRayScaleArray = 'u'
slice3Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice3Display.SelectOrientationVectors = 'vec'
slice3Display.ScaleFactor = 0.6250460354611278
slice3Display.SelectScaleArray = 'u'
slice3Display.GlyphType = 'Arrow'
slice3Display.GlyphTableIndexArray = 'u'
slice3Display.GaussianRadius = 0.031252301773056386
slice3Display.SetScaleArray = ['POINTS', 'u']
slice3Display.ScaleTransferFunction = 'PiecewiseFunction'
slice3Display.OpacityArray = ['POINTS', 'u']
slice3Display.OpacityTransferFunction = 'PiecewiseFunction'
slice3Display.DataAxesGrid = 'GridAxesRepresentation'
slice3Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
slice3Display.OSPRayScaleFunction.Points = [-6.608525935006737, 0.0, 0.5, 0.0, 50.675240851637405, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice3Display.ScaleTransferFunction.Points = [-0.1548475558938387, 0.0, 0.5, 0.0, 0.17817455546354832, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice3Display.OpacityTransferFunction.Points = [-0.1548475558938387, 0.0, 0.5, 0.0, 0.17817455546354832, 1.0, 0.5, 0.0]

# show data from gradientOfUnstructuredDataSet1
gradientOfUnstructuredDataSet1Display = Show(gradientOfUnstructuredDataSet1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
gradientOfUnstructuredDataSet1Display.Representation = 'Outline'
gradientOfUnstructuredDataSet1Display.ColorArrayName = ['POINTS', '']
gradientOfUnstructuredDataSet1Display.OSPRayScaleArray = 'u'
gradientOfUnstructuredDataSet1Display.OSPRayScaleFunction = 'PiecewiseFunction'
gradientOfUnstructuredDataSet1Display.SelectOrientationVectors = 'vec'
gradientOfUnstructuredDataSet1Display.ScaleFactor = 1.250092508
gradientOfUnstructuredDataSet1Display.SelectScaleArray = 'u'
gradientOfUnstructuredDataSet1Display.GlyphType = 'Arrow'
gradientOfUnstructuredDataSet1Display.GlyphTableIndexArray = 'u'
gradientOfUnstructuredDataSet1Display.GaussianRadius = 0.0625046254
gradientOfUnstructuredDataSet1Display.SetScaleArray = ['POINTS', 'u']
gradientOfUnstructuredDataSet1Display.ScaleTransferFunction = 'PiecewiseFunction'
gradientOfUnstructuredDataSet1Display.OpacityArray = ['POINTS', 'u']
gradientOfUnstructuredDataSet1Display.OpacityTransferFunction = 'PiecewiseFunction'
gradientOfUnstructuredDataSet1Display.DataAxesGrid = 'GridAxesRepresentation'
gradientOfUnstructuredDataSet1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
gradientOfUnstructuredDataSet1Display.OSPRayScaleFunction.Points = [-6.608525935006737, 0.0, 0.5, 0.0, 50.675240851637405, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
gradientOfUnstructuredDataSet1Display.ScaleTransferFunction.Points = [-0.2344534071814195, 0.0, 0.5, 0.0, 0.36425620750914156, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
gradientOfUnstructuredDataSet1Display.OpacityTransferFunction.Points = [-0.2344534071814195, 0.0, 0.5, 0.0, 0.36425620750914156, 1.0, 0.5, 0.0]

# show data from contour1
contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
contour1Display.Representation = 'Surface'
contour1Display.AmbientColor = [0.0, 1.0, 0.0]
contour1Display.ColorArrayName = ['POINTS', '']
contour1Display.DiffuseColor = [0.0, 1.0, 0.0]
contour1Display.OSPRayScaleArray = 'Q-criterion'
contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour1Display.SelectOrientationVectors = 'vec'
contour1Display.ScaleFactor = 1.2500925477594138
contour1Display.SelectScaleArray = 'Q-criterion'
contour1Display.GlyphType = 'Arrow'
contour1Display.GlyphTableIndexArray = 'Q-criterion'
contour1Display.GaussianRadius = 0.06250462738797069
contour1Display.SetScaleArray = ['POINTS', 'Q-criterion']
contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
contour1Display.OpacityArray = ['POINTS', 'Q-criterion']
contour1Display.OpacityTransferFunction = 'PiecewiseFunction'
contour1Display.DataAxesGrid = 'GridAxesRepresentation'
contour1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
contour1Display.OSPRayScaleFunction.Points = [-6.608525935006737, 0.0, 0.5, 0.0, 50.675240851637405, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour1Display.ScaleTransferFunction.Points = [-5.0, 0.0, 0.5, 0.0, -4.9990234375, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour1Display.OpacityTransferFunction.Points = [-5.0, 0.0, 0.5, 0.0, -4.9990234375, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for vLUT in view renderView1
vLUTColorBar = GetScalarBar(vLUT, renderView1)
vLUTColorBar.WindowLocation = 'UpperRightCorner'
vLUTColorBar.Title = 'v'
vLUTColorBar.ComponentTitle = ''

# set color bar visibility
vLUTColorBar.Visibility = 1

# show color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# show color legend
slice2Display.SetScalarBarVisibility(renderView1, True)

# show color legend
slice3Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'v'
vPWF = GetOpacityTransferFunction('v')
vPWF.Points = [-0.15, 0.0, 0.5, 0.0, 0.15, 1.0, 0.5, 0.0]
vPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# finally, restore active source
SetActiveSource(calculator1)
# ----------------------------------------------------------------
WriteImage("imagevQ.0425.png")

