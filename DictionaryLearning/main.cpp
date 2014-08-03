#include "CImg.h"
#include <mkl.h>

using namespace cimg_library;

struct PatchGroup
{
	double *data;
	unsigned int cnt;
};

struct ImagePair
{
	double *imgHR;
	double *imgLR;
	unsigned int width;
	unsigned int height;
	unsigned int colors;
};

struct Dictionary
{
	double *data;
	unsigned int numAtoms;
	unsigned int dictSize;
};

struct TrainParams
{
	unsigned int patchPerImage;
	unsigned int patchSize;
	unsigned int windowSize;
	double norm;
	double thresh;
	double recError;
	unsigned int sparsity;
};

int loadDict(Dictionary &dict, char *fname);
void randPatchTrain(ImagePair img, Dictionary dict, TrainParams params);
void vectorizePatch(double *img, double *vector, unsigned int x, unsigned int y, unsigned int c, unsigned int w, unsigned int h, unsigned int bsize);
double *groupPatches(double *imgLR, double *imgHR, double *patchGroup, double *dist, unsigned int *xPos, unsigned int *yPos, unsigned int ch, double thresh, double bsize, double wsize, unsigned int h, unsigned int w, unsigned int &cnt);
double patchDist(double *nPatch, double *tPatch, unsigned int bsize, double norm);
double *getSparseCoefficients(double *patchGroup, double *dict, unsigned int cnt, unsigned int gsize, double sparsity, double recError);
void updateDictionary(double *patchGroup, double *dict, double *gamma, unsigned int cnt, unsigned int gsize);
double getTime();


int main(int argc, char *argv[])
{
	int i, j;
	unsigned int w, h, c;
	double sTime, eTime;
	CImg<double> imageLR, imageHR;

	char *fpath, *fname;
	unsigned int numFiles, numEpochs;

	ImagePair img;
	Dictionary dict;
	TrainParams params;

	/*********************************************************
	Parameter list:
		1 - dictionary filename
		2 - path to image repository
		3 - number of images in repository
		4 - patches per image
		5 - number of epochs (passes through all images)
		6 - window size
		7 - norm											
		8 - target sparsity
		9 - max reconstruction error 
	**********************************************************/

	// Check if number of arguments is correct
	if (argc != 10)
	{
		printf("Incorrect number of parameters\n");
		return -1;
	}

	// Load the dictionary file
	if (loadDict(dict, argv[1]))
	
		return -1;

	// Copy the path to images
	fpath = argv[2];
	fname = (char *)malloc(sizeof(char)* (strlen(fpath) + 20));

	// Get the number of images in the repository
	numFiles = atoi(argv[3]);
	if (numFiles < 0)
	{
		printf("Invalid number of files in repository\n");
		return -1;
	}

	// Get the number of patches for each image
	params.patchPerImage = atoi(argv[4]);
	if (params.patchPerImage < 0)
	{
		printf("Invalid number of patches for each image\n");
		return -1;
	}

	// Get the number of epochs
	numEpochs = atoi(argv[5]);
	if (numEpochs < 0)
	{
		printf("Invalid number of epochs\n");
		return -1;
	}
	
	// Calculate the size of the patches
	params.patchSize = (unsigned int)sqrt((double)dict.dictSize / 5);
	if (dict.dictSize != (5 * params.patchSize * params.patchSize))
	{
		printf("Patches are non-square. Please check the provided dictionary file\n");
		return -1;
	}

	// Get the window size
	params.windowSize = atoi(argv[6]);
	if (params.windowSize < 0)
	{
		printf("Invalid window size\n");
		return -1;
	}
	else if (params.windowSize % 2 == 0)
	{
		printf("Increasing window size by 1 to make it odd\n");
		params.windowSize++;
	}

	// Get the norm
	params.norm = atof(argv[7]);

	// Get the target sparsity
	params.sparsity = atoi(argv[8]);
	if (params.sparsity < 0)
	{
		printf("Invalid target sparsity\n");
		return -1;
	}

	// Get the reconstruction error
	params.recError = atof(argv[9]);
	if (params.recError < 0)
	{
		printf("Invalid reconstruction error\n");
		return -1;
	}

	// Seed randomizer to 0
	srand(0);

	// Run the training for a certain number of epochs
	for (i = 0; i < numEpochs; i++)
	{
		// Train on each file inside the training repository
		for (j = 0; j < numFiles; j++)
		{
			// Complete the filename
			sprintf(fname, "%s\\%06d.png", fpath, j);

			// Load the image file
			imageHR.load(fname);

			// Extract image dimensions
			h = imageHR.height();
			w = imageHR.width();
			c = imageHR.spectrum();

			// Transpose the image to column-major format
			imageHR.transpose();

			// Generate a low resolution image
			imageLR = imageHR;
			imageLR.resize(h / 2, w / 2, 1, c, 6);

			// Copy the address of the image data to their respective pointers
			img.imgLR = imageLR.data();
			img.imgHR = imageHR.data();
			img.width = w / 2;
			img.height = h / 2;
			img.colors = c;

			// Perform random block matching on the image
			sTime = getTime();
			randPatchTrain(img, dict, params);
			eTime = getTime() - sTime;

			printf("Total time elapsed: %f sec\n", eTime);
		}
	}

	system("PAUSE");
	return 0;
}

int loadDict(Dictionary &dict, char *fname)
{
	unsigned int i, j, w, h;
	double tmp;
	FILE *fh;

	if (!(fh = fopen("train.dict", "r")))
	{
		printf("Error reading dictionary file\n");
		return -1;
	}

	// Read the dimensions of the dictionary
	fscanf(fh, "%d\n", &h);
	fscanf(fh, "%d\n", &w);

	// Allocate memory for the dictionary
	dict.data = (double *)mkl_malloc(sizeof(double)* h * w, 64);
	for (j = 0; j < h; j++)
	{
		for (i = 0; i < w; i++)
		{
			fscanf(fh, "%lf\n", &dict.data[h*j + i]);
		}
	}

	fclose(fh);

	dict.dictSize = h;
	dict.numAtoms = w;

	return 0;
}

void randPatchTrain(ImagePair img, Dictionary dict, TrainParams params)
{
	const int sqPatchSize = params.patchSize * params.patchSize;
	const int sqWindowSize = params.windowSize * params.windowSize;
	//const int gsize = 5 * bsize * bsize;
	const int halfWindowSize = (params.windowSize - 1) / 2;

	//int i, j;
	//unsigned int a, n, x, y, ch, cnt;
	unsigned int n, x, y, c;
	unsigned int *xPos, *yPos;

	//double tmp, mean;
	double *dist, *nPatch, *tPatch, *uPatch, *patchGroup;

	// Allocate memory for the normalized patches
	nPatch = (double *)mkl_malloc(sizeof(double)* sqPatchSize, 64);
	tPatch = (double *)mkl_malloc(sizeof(double)* sqPatchSize, 64);
	uPatch = (double *)mkl_malloc(sizeof(double)* sqPatchSize, 64);

	// Allocate memory for the distance and position matrices
	dist = (double *)mkl_malloc(sizeof(double)* sqWindowSize, 64);
	xPos = (unsigned int *)mkl_malloc(sizeof(unsigned int)* sqWindowSize, 64);
	yPos = (unsigned int *)mkl_malloc(sizeof(unsigned int)* sqWindowSize, 64);

	// Pre-allocate patch group
	patchGroup = (double *)mkl_malloc(sizeof(double),64);

	if (params.norm == 1)
	{
		for (n = 0; n < params.patchPerImage; n++)
		{
			// Select a random block from (hwindow, hwindow) to (h-hwindow-bsize-1, w-hwindow-bsize-1)
			x = (rand() % (img.width - params.windowSize - params.patchSize)) + halfWindowSize;
			y = (rand() % (img.height - params.windowSize - params.patchSize)) + halfWindowSize;
			c = rand() % img.colors;

			// Copy patch pixels into vector in column-major format
			vectorizePatch(imgLR, nPatch, x, y, ch, w, h, bsize);

			// Calculate the sum of absolute magnitudes (l1-norm) of the vector
			tmp = 1 / cblas_dasum(vsize, nPatch, 1);

			// Multiply the original vector by a scaling factor
			cblas_dscal(vsize, tmp, nPatch, 1);

			// Process each neighbor
			a = 0;
			for (i = -hwindow; i <= hwindow; i++)
			{
				for (j = -hwindow; j <= hwindow; j++)
				{
					// Copy patch pixels into vector in column-major format
					vectorizePatch(imgLR, tPatch, x+j, y+i, ch, w, h, bsize);

					// Calculate the sum of absolute magnitudes (l1-norm) of the vector
					tmp = 1 / cblas_dasum(vsize, tPatch, 1);

					// Multiply the original vector by a scaling factor
					cblas_dscal(vsize, tmp, tPatch, 1);

					// Calculate the patch distance
					vdsub(&vsize, nPatch, tPatch, uPatch);

					// Store the distance and position descriptors
					dist[a] = cblas_dasum(vsize, uPatch, 1);
					xPos[a] = x + j;
					yPos[a] = y + i;

					a++;
				}
			}

			// Group similar patches
			patchGroup = groupPatches(imgLR, imgHR, patchGroup, dist, xPos, yPos, ch, thresh, bsize, wsize, h, w, cnt);

			// Determine the sparse coefficients
			getSparseCoefficients(patchGroup, dict, cnt, gsize, sparsity, recError);
		}
	}
//	else if (norm == 2)
//	{
//		for (n = 0; n < RANDOM_PATCHES_PER_FILE; n++)
//		{
//			// Select a random block from (hwindow, hwindow) to (h-hwindow-bsize-1, w-hwindow-bsize-1)
//			// Generally, this means generating a random coordinate from 0 to <h/w>-window-bsize and adding hwindow
//			x = (rand() % (w - window - bsize)) + hwindow;
//			y = (rand() % (h - window - bsize)) + hwindow;
//			ch = rand() % c;
//
//			// Copy patch pixels into vector in column-major format
//			vectorizePatch(imgLR, nPatch, x, y, ch, w, h, bsize);
//
//			// Calculate the l2-norm of the vector
//			tmp = 1 / cblas_dnrm2(vsize, nPatch, 1);
//
//			// Multiply the original vector by a scaling factor
//			cblas_dscal(vsize, tmp, nPatch, 1);
//
//			// Process each neighbor
//			a = 0;
//			for (i = -hwindow; i <= hwindow; i++)
//			{
//				for (j = -hwindow; j <= hwindow; j++)
//				{
//					// Copy patch pixels into vector in column-major format
//					vectorizePatch(imgLR, tPatch, x+j, y+i, ch, w, h, bsize);
//
//					// Calculate the l2-norm of the vector
//					tmp = 1 / cblas_dnrm2(vsize, tPatch, 1);
//
//					// Multiply the original vector by a scaling factor
//					cblas_dscal(vsize, tmp, tPatch, 1);
//
//					// Calculate the patch distance
//					vdsub(&vsize, nPatch, tPatch, uPatch);
//
//					// Store the distance and position descriptors
//					dist[a] = cblas_dnrm2(vsize, uPatch, 1);
//					xPos[a] = x + j;
//					yPos[a] = y + i;
//
//					a++;
//				}
//			}
//
//			// Group similar patches
//			patchGroup = groupPatches(imgLR, imgHR, patchGroup, dist, xPos, yPos, ch, thresh, bsize, wsize, h, w, cnt);
//		}
//	}
//	else if (norm > 0)
//	{
//		for (n = 0; n < RANDOM_PATCHES_PER_FILE; n++)
//		{
//			// Select a random block from (hwindow, hwindow) to (h-hwindow-bsize-1, w-hwindow-bsize-1)
//			// Generally, this means generating a random coordinate from 0 to <h/w>-window-bsize and adding hwindow
//			x = (rand() % (w - window - bsize)) + hwindow;
//			y = (rand() % (h - window - bsize)) + hwindow;
//			ch = rand() % c;
//
//			// Copy patch pixels into vector in column-major format
//			vectorizePatch(imgLR, nPatch, x, y, ch, w, h, bsize);
//
//			// Calculate the lp-norm of the vector
//			vdabs(&vsize, nPatch, tPatch);
//			vdpowx(&vsize, tPatch, &norm, tPatch);
//			tmp = 1 / pow(cblas_dasum(vsize, tPatch, 1), 1/norm);
//
//			// Multiply the original vector by a scaling factor
//			cblas_dscal(vsize, tmp, nPatch, 1);
//
//			// Process each neighbor
//			a = 0;
//			for (i = -hwindow; i <= hwindow; i++)
//			{
//				for (j = -hwindow; j <= hwindow; j++)
//				{
//					// Copy patch pixels into vector in column-major format
//					vectorizePatch(imgLR, tPatch, x+j, y+i, ch, w, h, bsize);
//
//					// Calculate the lp-norm of the vector
//					vdabs(&vsize, tPatch, uPatch);
//					vdpowx(&vsize, uPatch, &norm, uPatch);
//					tmp = 1 / pow(cblas_dasum(vsize, uPatch, 1), 1/norm);
//
//					// Multiply the original vector by a scaling factor
//					cblas_dscal(vsize, tmp, tPatch, 1);
//
//					// Calculate the patch distance
//					vdsub(&vsize, nPatch, tPatch, uPatch);
//					vdabs(&vsize, uPatch, uPatch);
//					vdpowx(&vsize, uPatch, &norm, uPatch);
//
//					// Store the distance and position descriptors
//					dist[a] = cblas_dasum(vsize, uPatch, 1);
//					xPos[a] = x + j;
//					yPos[a] = y + i;
//
//					a++;
//				}
//			}
//
//			// Group similar patches
//			patchGroup = groupPatches(imgLR, imgHR, patchGroup, dist, xPos, yPos, ch, thresh, bsize, wsize, h, w, cnt);
//		}
//	}
//	else
//	{
//		for (n = 0; n < RANDOM_PATCHES_PER_FILE; n++)
//		{
//			// Select a random block from (hwindow, hwindow) to (h-hwindow-bsize-1, w-hwindow-bsize-1)
//			// Generally, this means generating a random coordinate from 0 to <h/w>-window-bsize and adding hwindow
//			x = (rand() % (w - window - bsize)) + hwindow;
//			y = (rand() % (h - window - bsize)) + hwindow;
//			ch = rand() % c;
//
//			// Copy patch pixels into vector in column-major format
//			vectorizePatch(imgLR, nPatch, x, y, ch, w, h, bsize);
//
//			// Process each neighbor
//			a = 0;
//			for (i = -hwindow; i <= hwindow; i++)
//			{
//				for (j = -hwindow; j <= hwindow; j++)
//				{
//					// Copy patch pixels into vector in column-major format
//					vectorizePatch(imgLR, tPatch, x+j, y+i, ch, w, h, bsize);
//
//					// Calculate the patch distance
//					vdsub(&vsize, nPatch, tPatch, uPatch);
//
//					// Store the distance and position descriptors
//					dist[a] = cblas_dnrm2(vsize, uPatch, 1);
//					xPos[a] = x + j;
//					yPos[a] = y + i;
//
//					a++;
//				}
//			}
//
//			// Group similar patches
//			patchGroup = groupPatches(imgLR, imgHR, patchGroup, dist, xPos, yPos, ch, thresh, bsize, wsize, h, w, cnt);
//		}
//	}
}

//void vectorizePatch(double *img, double *vector, unsigned int x, unsigned int y, unsigned int c, unsigned int w, unsigned int h, unsigned int bsize)
//{
//	const int tsize = bsize;
//	const int inc = 1;
//	double mean, tmp;
//	unsigned int i, j, a;
//	int p, n, xstorage;
//
//	// Copy the patch pixels to the vector in column-major format
//	a = 0;
//	mean = 0;
//	for (j = 0; j < bsize; j++)
//	{
//		cblas_dcopy(bsize, &img[h*w*c+h*(x+j)+y], 1, &vector[a], 1);
//		for (i = 0; i < bsize; i++)
//		{
//			mean += img[h*w*c+h*(x+j)+y+i];
//		}
//		a += bsize;
//	}
//	mean /= (bsize * bsize);
//
//	for (a = 0; a < bsize * bsize; a++)
//	{
//		vector[a] -= mean;
//	}
//
//	return;
//}
//
//double *groupPatches(double *imgLR, double *imgHR, double *patchGroup, double *dist, unsigned int *xPos, unsigned int *yPos, unsigned int ch, double thresh, double bsize, double wsize, unsigned int h, unsigned int w, unsigned int &cnt)
//{
//	const int lrsize = bsize;
//	const int hrsize = 2 * bsize;
//	const int vsize = bsize * bsize;
//	const int gsize = 5 * bsize * bsize;
//
//	int i, j;
//	unsigned int a, b, t;
//
//	// Count the number of blocks with distances falling below the threshold
//	cnt = 0;
//	for (a = 0; a < wsize; a++)
//	{
//		if (dist[a] < thresh)
//		{
//			cnt++;
//		}
//	}
//
//	// Allocate a matrix for the patch group
//	patchGroup = (double *)mkl_realloc(patchGroup, sizeof(double)* gsize * cnt);
//
//	// Populate the matrix
//	t = 0;
//	for (a = 0; a < wsize; a++)
//	{
//		if (dist[a] < thresh)
//		{
//			// Reset the row counter
//			b = 0;
//
//			// Store the LR patch
//			for (j = 0; j < lrsize; j++)
//			{
//				cblas_dcopy(lrsize, &imgLR[h*w*ch+h*(xPos[a]+j)+yPos[a]], 1, &patchGroup[gsize*t+b], 1);
//				b += lrsize;
//			}
//
//			// Store the HR patch
//			for (j = 0; j < hrsize; j++)
//			{
//				cblas_dcopy(hrsize, &imgHR[4*h*w*ch+2*h*(2*xPos[a]+j)+2*yPos[a]], 1, &patchGroup[gsize*t+b], 1);
//				b += hrsize;
//			}
//
//			t++;
//		}
//	}
//
//	return patchGroup;
//}
//
//double *getSparseCoefficients(double *patchGroup, double *dict, unsigned int cnt, unsigned int gsize, double sparsity, double recError)
//{
//	const int noa = NUMBER_OF_ATOMS;
//	const int numel = NUMBER_OF_ATOMS * cnt;
//	const int npix = gsize * cnt;
//
//	int i;
//	unsigned int n, ind;
//	unsigned int *list;
//	double tmp;
//	double *alpha, *talpha, *alphaTotal, *gamma, *residual, *signal, *tdict, *corr, *L, *L2, *T;
//
//	list = (unsigned int *)mkl_malloc(sizeof(unsigned int) * NUMBER_OF_ATOMS, 64);
//
//	// Allocate necessary resources
//	L = (double *)mkl_malloc(sizeof(double), 64);
//	L2 = (double *)mkl_malloc(sizeof(double), 64);
//	gamma = (double *)mkl_malloc(sizeof(double) * NUMBER_OF_ATOMS * cnt, 64);
//	alpha = (double *)mkl_malloc(sizeof(double) * NUMBER_OF_ATOMS * cnt, 64);
//	talpha = (double *)mkl_malloc(sizeof(double) * NUMBER_OF_ATOMS * cnt, 64);
//	alphaTotal = (double *)mkl_malloc(sizeof(double) * NUMBER_OF_ATOMS, 64);
//	residual = (double *)mkl_malloc(sizeof(double) * gsize * cnt, 64);
//	signal = (double *)mkl_malloc(sizeof(double) * gsize * cnt, 64);
//	tdict = (double *)mkl_malloc(sizeof(double) * gsize * NUMBER_OF_ATOMS, 64);
//	corr = (double *)mkl_malloc(sizeof(double) * NUMBER_OF_ATOMS, 64);
//
//	// Find the correlation between the signals and the dictionary
//	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, NUMBER_OF_ATOMS, cnt, gsize, 1, dict, gsize, patchGroup, gsize, 0, alpha, NUMBER_OF_ATOMS);
//	vdabs(&numel, alpha, talpha);
//
//	// Compute for the TOTAL correlation of the dictionary with the patches
//	cblas_dscal(NUMBER_OF_ATOMS, 0, alphaTotal, 1);
//	for (i = 0; i < cnt; i++)
//	{
//		vdadd(&noa, &talpha[NUMBER_OF_ATOMS*i], alphaTotal, alphaTotal);
//	}
//
//	// Find the largest correlation and consider it the first sparse coefficient (gamma)
//	L[0] = 1;
//	list[0] = cblas_idamax(NUMBER_OF_ATOMS, alphaTotal, 1);
//	cblas_dcopy(cnt, &alpha[list[0]], NUMBER_OF_ATOMS, talpha, 1);
//
//	// Copy the dictionary atom to the temporary dictionary
//	cblas_dcopy(gsize, &dict[gsize*list[0]], 1, tdict, 1); 
//	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, gsize, cnt, 1, 1, tdict, gsize, talpha, 1, 0, signal, gsize);
//
//	// Compute for the resulting signal using the selected atom
//	vdsub(&npix, patchGroup, signal, residual);
//
//	n = 1;
//	do
//	{
//		// Find the correlation between the signals and the dictionary
//		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, NUMBER_OF_ATOMS, cnt, gsize, 1, dict, gsize, residual, gsize, 0, talpha, NUMBER_OF_ATOMS);
//		vdabs(&numel, talpha, talpha);
//
//		// Compute for the TOTAL correlation of the dictionary with the patches
//		cblas_dscal(NUMBER_OF_ATOMS, 0, alphaTotal, 1);
//		for (i = 0; i < cnt; i++)
//		{
//			vdadd(&noa, &talpha[NUMBER_OF_ATOMS*i], alphaTotal, alphaTotal);
//		}
//
//		// Find the largest correlation
//		list[n] = cblas_idamax(NUMBER_OF_ATOMS, alphaTotal, 1);
//
//		// D'd_k
//		cblas_dgemv(CblasColMajor, CblasTrans, gsize, n, 1, tdict, gsize, &dict[gsize*list[n]], 1, 0, corr, 1);
//		cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, L, n, corr, 1);
//		tmp = cblas_dnrm2(n, corr, 1);
//
//		// Re-allocate memory for the lower triangular matrices and copy the old matrix contents
//		L2 = (double *)mkl_realloc(L2, sizeof(double) * (n+1) * (n+1));
//		for (i = 0; i < n; i++)
//		{
//			// Copy the old column of L to L2
//			cblas_dcopy(n, &L[n*i], 1, &L2[(n+1)*i], 1);
//
//			// To save on loops, we recycle this loop for alpha selection as well
//			// Copy the rows of alpha which are needed to solve for gamma later
//			cblas_dcopy(cnt, &alpha[list[i]], NUMBER_OF_ATOMS, &talpha[i], n+1);
//		}
//
//		// Copy the last alpha row
//		cblas_dcopy(cnt, &alpha[list[n]], NUMBER_OF_ATOMS, &talpha[n], n+1);
//
//		// Zero out the last column of L2
//		cblas_dscal(n, 0, &L2[(n+1)*n], 1);
//
//		// Add the last row to L2
//		cblas_dcopy(n, corr, 1, &L2[n], n+1);
//		L2[(n+1)*n+n] = sqrt(1 - tmp * tmp);
//
//		// Swap matrix pointers
//		T = L;
//		L = L2;
//		L2 = T;
//
//		// Solve for the sparse coefficients
//		cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n+1, cnt, 1, L, n+1, talpha, n+1);
//		cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, n+1, cnt, 1, L, n+1, talpha, n+1);
//
//		// Copy the dictionary atom to the temporary dictionary
//		cblas_dcopy(gsize, &dict[gsize*list[n]], 1, &tdict[gsize*n], 1); 
//		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, gsize, cnt, n+1, 1, tdict, gsize, talpha, n+1, 0, signal, gsize);
//
//		// Compute for the resulting signal using the selected atom
//		vdsub(&npix, patchGroup, signal, residual);
//
//		// Calculate the error
//		tmp = cblas_dnrm2(npix, residual, 1) / sqrt((double)cnt);
//
//		n++;
//	} while (tmp > recError && n < sparsity);
//
//	// Populate the final gamma
//	cblas_dscal(numel, 0, gamma, 1);
//	for (i = 0; i < n; i++)
//	{
//		cblas_dcopy(cnt, &talpha[i], n, &gamma[list[i]], NUMBER_OF_ATOMS);
//	}
//
//	//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, gsize, cnt, NUMBER_OF_ATOMS, 1, dict, gsize, gamma, NUMBER_OF_ATOMS, 0, residual, gsize);
//
//	//CImg<double> imA, imB, imC, imZ(1, BLOCK_SIZE, 1, 1);
//	//imA.assign(patchGroup, BLOCK_SIZE, BLOCK_SIZE, 1, 1);
//	//imB.assign(signal, BLOCK_SIZE, BLOCK_SIZE, 1, 1);
//	////imC.assign(residual, BLOCK_SIZE, BLOCK_SIZE, 1, 1);
//	//imZ.fill(0);
//	//imA.append(imZ);
//	//imA.append(imB);
//	////imA.append(imZ);
//	////imA.append(imC);
//	//imA.display();
//
//	mkl_free(list);
//	mkl_free(L);
//	mkl_free(L2);
//	mkl_free(alpha);
//	mkl_free(talpha);
//	mkl_free(alphaTotal);
//	mkl_free(residual);
//	mkl_free(signal);
//	mkl_free(tdict);
//	mkl_free(corr);
//
//	return gamma;
//}
//
//void updateDictionary(double *patchGroup, double *dict, double *gamma, unsigned int cnt, unsigned int gsize, double *M, double *C)
//{
//	int n, i, j;
//	double *tM, *tC, *w;
//
//	tM = (double *)mkl_malloc(sizeof(double) * NUMBER_OF_ATOMS * NUMBER_OF_ATOMS, 64);
//	tC = (double *)mkl_malloc(sizeof(double) * NUMBER_OF_ATOMS, 64);
//	w = (double *)mkl_malloc(sizeof(double) * NUMBER_OF_ATOMS * cnt, 64);
//
//	for (n = 0; n < 10; n++)
//	{
//		for (j = 0; j < gsize; j++)
//		{
//			for (i = 0; i < cnt; i++)
//			{
//
//			}
//		}
//	}
//}

double getTime()
{
	LARGE_INTEGER t, f;
	QueryPerformanceCounter(&t);
	QueryPerformanceFrequency(&f);
	return (double)t.QuadPart / (double)f.QuadPart;
}

