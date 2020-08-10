import pulse2percept as p2p
from glob import glob
import sys
import numpy as np
from os.path import split, splitext, join

def main():
    if len(sys.argv) < 3:
        print('Usage: ./run-smm-images.py <path-to-files> <model> <preprocess> <rho=100> <lambda=100>')
        return

    # Model string
    pathstr = sys.argv[1]
    modelstr = sys.argv[2]
    preprocess = np.int(sys.argv[3])
    prestr = '-pre' if preprocess else ''
    
    rho = int(sys.argv[4]) if len(sys.argv) >= 5 else 100
    rhostr = 'rho%d-' % rho
    axlambda = int(sys.argv[5]) if len(sys.argv) >= 6 else 100
    lmbstr = ('lmb%d-' % axlambda) if modelstr == 'axonmap' else ''
    
    models = {
        'scoreboard': p2p.models.ScoreboardModel,
        'axonmap': p2p.models.AxonMapModel
    }
    
    # Set up the axon map model at a reasonable step size.
    # Might want to play with different values for rho/lambda:
    model = models[modelstr](xrange=(-25, 15), yrange=(-15, 15), xystep=0.5,
                             engine='serial', rho=rho)
    if modelstr == 'axonmap':
        model.build(axlambda=axlambda, n_axons=1500)
    else:
        model.build()
    
    for file in glob(join(pathstr, '*.jpg')):
        print(file)
        # For each grid, generate the percept and store it in the list:
        path, fname = split(file)
        fstr, fext = splitext(fname)
        img = p2p.stimuli.ImageStimulus(file, as_gray=True)
        for n in np.arange(2, 18):
            gsize = (4 * n, 6 * n)
            # Make all implants span the same area on the retina, so we need to adjust
            # the electrode-to-electrode spacing depending on how many rows/columns in
            # the grid:
            spacing = 10000.0 / gsize[1]
            grid = p2p.implants.ElectrodeGrid(gsize, x=-1500, spacing=spacing, 
                                              etype=p2p.implants.DiskElectrode, r=100)
            implant = p2p.implants.ProsthesisSystem(grid)
            if preprocess:
                implant.stim = img.filter('median').filter('sobel').resize(gsize)
            else:
                implant.stim = img.resize(gsize)
            percept = model.predict_percept(implant)
            pfname = join(path, 'percepts', '%s-%s%s-%s%s%dx%d%s' % (fstr, modelstr, prestr, rhostr, lmbstr, *gsize, fext))
            percept.save(pfname, shape=img.img_shape)
            

if __name__ == "__main__":
    main()