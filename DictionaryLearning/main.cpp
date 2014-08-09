#include "CImg.h"
#include <mkl.h>

using namespace cimg_library;

struct PatchGroup
{
	double *data;
	unsigned int cnt;
};

struct PatchDesc
{
	double *dist;
	unsigned int *xPos;
	unsigned int *yPos;
	unsigned int channel;
};

struct ImagePair
{
	double *imgHR;
	double *imgLR;
	unsigned int width;
	unsigned int height;
	unsigned int channels;
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
void groupPatches(ImagePair img, PatchDesc desc, TrainParams params, PatchGroup &grp);
double patchDist(double *nPatch, double *tPatch, unsigned int bsize, double norm);
void getSparseCoefficients(PatchGroup grp, Dictionary dict, TrainParams params);
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
		0  - name of executable file (not needed)
		1  - dictionary filename
		2  - path to image repository
		3  - number of images in repository
		4  - patches per image
		5  - number of epochs (passes through all images)
		6  - window size
		7  - norm											
		8  - target sparsity
		9  - max reconstruction error
		10 - distance threshold between patches
	**********************************************************/

	// Check if number of arguments is correct
	if (argc != 11)
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

	// Get the distance threshold between patches
	params.thresh = atof(argv[10]);
	if (params.thresh < 0)
	{
		printf("Invalid patch distance threshold\n");
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
			img.channels = c;

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

	int i, j;
	//unsigned int a, n, x, y, ch, cnt;
	unsigned int a, n, x, y, c;

	double tmp;
	double *nPatch, *tPatch, *uPatch;

	PatchDesc desc;
	PatchGroup grp;

	// Allocate memory for the normalized patches
	nPatch = (double *)mkl_malloc(sizeof(double)* sqPatchSize, 64);
	tPatch = (double *)mkl_malloc(sizeof(double)* sqPatchSize, 64);
	uPatch = (double *)mkl_malloc(sizeof(double)* sqPatchSize, 64);

	// Allocate memory for the distance and position matrices
	desc.dist = (double *)mkl_malloc(sizeof(double)* sqWindowSize, 64);
	desc.xPos = (unsigned int *)mkl_malloc(sizeof(unsigned int)* sqWindowSize, 64);
	desc.yPos = (unsigned int *)mkl_malloc(sizeof(unsigned int)* sqWindowSize, 64);

	// Pre-allocate patch group
	grp.data = (double *)mkl_malloc(sizeof(double) * dict.dictSize * sqWindowSize,64);

	if (params.norm == 1)
	{
		for (n = 0; n < params.patchPerImage; n++)
		{
			// Select a random block from (halfWindowSize, halfWindowSize) to (h-halfWindowSize-bsize-1, w-halfWindowSize-bsize-1)
			x = (rand() % (img.width - params.windowSize - params.patchSize)) + halfWindowSize;
			y = (rand() % (img.height - params.windowSize - params.patchSize)) + halfWindowSize;
			c = rand() % img.channels;

			// Copy patch pixels into vector in column-major format
			vectorizePatch(img.imgLR, nPatch, x, y, c, img.width, img.height, params.patchSize);

			// Calculate the sum of absolute magnitudes (l1-norm) of the vector
			tmp = 1 / cblas_dasum(sqPatchSize, nPatch, 1);

			// Multiply the original vector by a scaling factor
			cblas_dscal(sqPatchSize, tmp, nPatch, 1);

			// Process each neighbor
			a = 0;
			for (i = -halfWindowSize; i <= halfWindowSize; i++)
			{
				for (j = -halfWindowSize; j <= halfWindowSize; j++)
				{
					// Copy patch pixels into vector in column-major format
					vectorizePatch(img.imgLR, tPatch, x+j, y+i, c, img.width, img.height, params.patchSize);

					// Calculate the sum of absolute magnitudes (l1-norm) of the vector
					tmp = 1 / cblas_dasum(sqPatchSize, tPatch, 1);

					// Multiply the original vector by a scaling factor
					cblas_dscal(sqPatchSize, tmp, tPatch, 1);

					// Calculate the patch distance
					vdsub(&sqPatchSize, nPatch, tPatch, uPatch);

					// Store the distance and position descriptors
					desc.dist[a] = cblas_dasum(sqPatchSize, uPatch, 1);
					desc.xPos[a] = x + j;
					desc.yPos[a] = y + i;

					a++;
				}
			}
			desc.channel = c;

			// Group similar patches
			groupPatches(img, desc, params, grp);

			// Determine the sparse coefficients
			getSparseCoefficients(grp, dict, params);
		}
	}
//	else if (norm == 2)
//	{
//		for (n = 0; n < RANDOM_PATCHES_PER_FILE; n++)
//		{
//			// Select a random block from (halfWindowSize, halfWindowSize) to (h-halfWindowSize-bsize-1, w-halfWindowSize-bsize-1)
//			// Generally, this means generating a random coordinate from 0 to <h/w>-window-bsize and adding halfWindowSize
//			x = (rand() % (w - window - bsize)) + halfWindowSize;
//			y = (rand() % (h - window - bsize)) + halfWindowSize;
//			ch = rand() % c;
//
//			// Copy patch pixels into vector in column-major format
//			vectorizePatch(imgLR, nPatch, x, y, ch, w, h, bsize);
//
//			// Calculate the l2-norm of the vector
//			tmp = 1 / cblas_dnrm2(sqPatchSize, nPatch, 1);
//
//			// Multiply the original vector by a scaling factor
//			cblas_dscal(sqPatchSize, tmp, nPatch, 1);
//
//			// Process each neighbor
//			a = 0;
//			for (i = -halfWindowSize; i <= halfWindowSize; i++)
//			{
//				for (j = -halfWindowSize; j <= halfWindowSize; j++)
//				{
//					// Copy patch pixels into vector in column-major format
//					vectorizePatch(imgLR, tPatch, x+j, y+i, ch, w, h, bsize);
//
//					// Calculate the l2-norm of the vector
//					tmp = 1 / cblas_dnrm2(sqPatchSize, tPatch, 1);
//
//					// Multiply the original vector by a scaling factor
//					cblas_dscal(sqPatchSize, tmp, tPatch, 1);
//
//					// Calculate the patch distance
//					vdsub(&sqPatchSize, nPatch, tPatch, uPatch);
//
//					// Store the distance and position descriptors
//					dist[a] = cblas_dnrm2(sqPatchSize, uPatch, 1);
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
//			// Select a random block from (halfWindowSize, halfWindowSize) to (h-halfWindowSize-bsize-1, w-halfWindowSize-bsize-1)
//			// Generally, this means generating a random coordinate from 0 to <h/w>-window-bsize and adding halfWindowSize
//			x = (rand() % (w - window - bsize)) + halfWindowSize;
//			y = (rand() % (h - window - bsize)) + halfWindowSize;
//			ch = rand() % c;
//
//			// Copy patch pixels into vector in column-major format
//			vectorizePatch(imgLR, nPatch, x, y, ch, w, h, bsize);
//
//			// Calculate the lp-norm of the vector
//			vdabs(&sqPatchSize, nPatch, tPatch);
//			vdpowx(&sqPatchSize, tPatch, &norm, tPatch);
//			tmp = 1 / pow(cblas_dasum(sqPatchSize, tPatch, 1), 1/norm);
//
//			// Multiply the original vector by a scaling factor
//			cblas_dscal(sqPatchSize, tmp, nPatch, 1);
//
//			// Process each neighbor
//			a = 0;
//			for (i = -halfWindowSize; i <= halfWindowSize; i++)
//			{
//				for (j = -halfWindowSize; j <= halfWindowSize; j++)
//				{
//					// Copy patch pixels into vector in column-major format
//					vectorizePatch(imgLR, tPatch, x+j, y+i, ch, w, h, bsize);
//
//					// Calculate the lp-norm of the vector
//					vdabs(&sqPatchSize, tPatch, uPatch);
//					vdpowx(&sqPatchSize, uPatch, &norm, uPatch);
//					tmp = 1 / pow(cblas_dasum(sqPatchSize, uPatch, 1), 1/norm);
//
//					// Multiply the original vector by a scaling factor
//					cblas_dscal(sqPatchSize, tmp, tPatch, 1);
//
//					// Calculate the patch distance
//					vdsub(&sqPatchSize, nPatch, tPatch, uPatch);
//					vdabs(&sqPatchSize, uPatch, uPatch);
//					vdpowx(&sqPatchSize, uPatch, &norm, uPatch);
//
//					// Store the distance and position descriptors
//					dist[a] = cblas_dasum(sqPatchSize, uPatch, 1);
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
//			// Select a random block from (halfWindowSize, halfWindowSize) to (h-halfWindowSize-bsize-1, w-halfWindowSize-bsize-1)
//			// Generally, this means generating a random coordinate from 0 to <h/w>-window-bsize and adding halfWindowSize
//			x = (rand() % (w - window - bsize)) + halfWindowSize;
//			y = (rand() % (h - window - bsize)) + halfWindowSize;
//			ch = rand() % c;
//
//			// Copy patch pixels into vector in column-major format
//			vectorizePatch(imgLR, nPatch, x, y, ch, w, h, bsize);
//
//			// Process each neighbor
//			a = 0;
//			for (i = -halfWindowSize; i <= halfWindowSize; i++)
//			{
//				for (j = -halfWindowSize; j <= halfWindowSize; j++)
//				{
//					// Copy patch pixels into vector in column-major format
//					vectorizePatch(imgLR, tPatch, x+j, y+i, ch, w, h, bsize);
//
//					// Calculate the patch distance
//					vdsub(&sqPatchSize, nPatch, tPatch, uPatch);
//
//					// Store the distance and position descriptors
//					dist[a] = cblas_dnrm2(sqPatchSize, uPatch, 1);
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

void vectorizePatch(double *img, double *vector, unsigned int x, unsigned int y, unsigned int c, unsigned int w, unsigned int h, unsigned int bsize)
{
	double mean, tmp;
	unsigned int i, j, a;

	// Copy the patch pixels to the vector in column-major format
	a = 0;
	mean = 0;
	for (j = 0; j < bsize; j++)
	{
		cblas_dcopy(bsize, &img[h*w*c+h*(x+j)+y], 1, &vector[a], 1);
		for (i = 0; i < bsize; i++)
		{
			mean += img[h*w*c+h*(x+j)+y+i];
		}
		a += bsize;
	}
	mean /= (bsize * bsize);

	for (a = 0; a < bsize * bsize; a++)
	{
		vector[a] -= mean;
	}

	return;
}

void groupPatches(ImagePair img, PatchDesc desc, TrainParams params, PatchGroup &grp)
{
	const int lrsize = params.patchSize;
	const int hrsize = 2 * params.patchSize;
	const int sqPatchSize = lrsize * lrsize;
	const int groupSize = 5 * sqPatchSize;
	const int sqWindowSize = params.windowSize * params.windowSize;

	int i, j;
	unsigned int a, b, t;

	// Count the number of blocks with distances falling below the threshold
	grp.cnt = 0;
	for (a = 0; a < sqWindowSize; a++)
	{
		if (desc.dist[a] < params.thresh)
		{
			grp.cnt++;
		}
	}

	// Populate the matrix
	t = 0;
	for (a = 0; a < sqWindowSize; a++)
	{
		if (desc.dist[a] < params.thresh)
		{
			// Reset the row counter
			b = 0;

			// Store the LR patch
			for (j = 0; j < lrsize; j++)
			{
				cblas_dcopy(lrsize, &img.imgLR[img.height*img.width*desc.channel+img.height*(desc.xPos[a]+j)+desc.yPos[a]], 1, &grp.data[groupSize*t+b], 1);
				b += lrsize;
			}

			// Store the HR patch
			for (j = 0; j < hrsize; j++)
			{
				cblas_dcopy(hrsize, &img.imgHR[4*img.height*img.width*desc.channel+2*img.height*(2*desc.xPos[a]+j)+2*desc.yPos[a]], 1, &grp.data[groupSize*t+b], 1);
				b += hrsize;
			}

			t++;
		}
	}
}

void getSparseCoefficients(PatchGroup grp, Dictionary dict, TrainParams params)
{
	const int dictSize = dict.dictSize;
	const int numAtoms = dict.numAtoms;
	const int numData = dict.numAtoms * grp.cnt;
	const int numPixel = dictSize * grp.cnt;

	int i;
	unsigned int n, ind;
	unsigned int *list;
	double tmp;
	double *alpha, *talpha, *alphaTotal, *gamma, *residual, *signal, *tdict, *corr, *L, *L2, *T;

	list = (unsigned int *)mkl_malloc(sizeof(unsigned int) * numAtoms, 64);

	// Allocate necessary resources
	L = (double *)mkl_malloc(sizeof(double) * numAtoms * numAtoms, 64);
	L2 = (double *)mkl_malloc(sizeof(double)* numAtoms * numAtoms, 64);
	gamma = (double *)mkl_malloc(sizeof(double) * numAtoms * grp.cnt, 64);
	alpha = (double *)mkl_malloc(sizeof(double)* numAtoms * grp.cnt, 64);
	talpha = (double *)mkl_malloc(sizeof(double)* numAtoms * grp.cnt, 64);
	alphaTotal = (double *)mkl_malloc(sizeof(double) * numAtoms, 64);
	residual = (double *)mkl_malloc(sizeof(double) * dictSize * grp.cnt, 64);
	signal = (double *)mkl_malloc(sizeof(double) * dictSize * grp.cnt, 64);
	tdict = (double *)mkl_malloc(sizeof(double) * dictSize * numAtoms, 64);
	corr = (double *)mkl_malloc(sizeof(double) * numAtoms, 64);

	// Find the correlation between the signals and the dictionary
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, numAtoms, grp.cnt, dictSize, 1, dict.data, dictSize, grp.data, dictSize, 0, alpha, numAtoms);
	vdabs(&numData, alpha, talpha);

	// Compute for the TOTAL correlation of the dictionary with the patches
	cblas_dscal(numAtoms, 0, alphaTotal, 1);
	for (i = 0; i < grp.cnt; i++)
	{
		vdadd(&numAtoms, &talpha[numAtoms*i], alphaTotal, alphaTotal);
	}

	// Find the largest correlation and consider it the first sparse coefficient (gamma)
	L[0] = 1;
	list[0] = cblas_idamax(numAtoms, alphaTotal, 1);
	cblas_dcopy(grp.cnt, &alpha[list[0]], numAtoms, talpha, 1);

	// Copy the dictionary atom to the temporary dictionary
	cblas_dcopy(dictSize, &dict.data[dictSize*list[0]], 1, tdict, 1); 
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dictSize, grp.cnt, 1, 1, tdict, dictSize, talpha, 1, 0, signal, dictSize);

	// Compute for the resulting signal using the selected atom
	vdsub(&numPixel, grp.data, signal, residual);

	n = 1;
	do
	{
		// Find the correlation between the signals and the dictionary
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, numAtoms, grp.cnt, dictSize, 1, dict.data, dictSize, residual, dictSize, 0, talpha, numAtoms);
		vdabs(&numData, talpha, talpha);

		// Compute for the TOTAL correlation of the dictionary with the patches
		cblas_dscal(numAtoms, 0, alphaTotal, 1);
		for (i = 0; i < grp.cnt; i++)
		{
			vdadd(&numAtoms, &talpha[numAtoms*i], alphaTotal, alphaTotal);
		}

		// Find the largest correlation
		list[n] = cblas_idamax(numAtoms, alphaTotal, 1);

		// D'd_k
		cblas_dgemv(CblasColMajor, CblasTrans, dictSize, n, 1, tdict, dictSize, &dict.data[dictSize*list[n]], 1, 0, corr, 1);
		cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, L, n, corr, 1);
		tmp = cblas_dnrm2(n, corr, 1);

		// Re-allocate memory for the lower triangular matrices and copy the old matrix contents
		L2 = (double *)mkl_realloc(L2, sizeof(double) * (n+1) * (n+1));
		for (i = 0; i < n; i++)
		{
			// Copy the old column of L to L2
			cblas_dcopy(n, &L[n*i], 1, &L2[(n+1)*i], 1);

			// To save on loops, we recycle this loop for alpha selection as well
			// Copy the rows of alpha which are needed to solve for gamma later
			cblas_dcopy(grp.cnt, &alpha[list[i]], numAtoms, &talpha[i], n+1);
		}

		// Copy the last alpha row
		cblas_dcopy(grp.cnt, &alpha[list[n]], numAtoms, &talpha[n], n+1);

		// Zero out the last column of L2
		cblas_dscal(n, 0, &L2[(n+1)*n], 1);

		// Add the last row to L2
		cblas_dcopy(n, corr, 1, &L2[n], n+1);
		L2[(n+1)*n+n] = sqrt(1 - tmp * tmp);

		// Swap matrix pointers
		T = L;
		L = L2;
		L2 = T;

		// Solve for the sparse coefficients
		cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n+1, grp.cnt, 1, L, n+1, talpha, n+1);
		cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, n+1, grp.cnt, 1, L, n+1, talpha, n+1);

		// Copy the dictionary atom to the temporary dictionary
		cblas_dcopy(dictSize, &dict.data[dictSize*list[n]], 1, &tdict[dictSize*n], 1); 
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dictSize, grp.cnt, n+1, 1, tdict, dictSize, talpha, n+1, 0, signal, dictSize);

		// Compute for the resulting signal using the selected atom
		vdsub(&numPixel, grp.data, signal, residual);

		// Calculate the error
		tmp = cblas_dnrm2(numPixel, residual, 1) / sqrt((double)grp.cnt);

		n++;
	} while (tmp > params.recError && n < params.sparsity);

	// Populate the final gamma
	cblas_dscal(numData, 0, gamma, 1);
	for (i = 0; i < n; i++)
	{
		cblas_dcopy(grp.cnt, &talpha[i], n, &gamma[list[i]], numAtoms);
	}

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dictSize, grp.cnt, numAtoms, 1, dict.data, dictSize, gamma, numAtoms, 0, residual, dictSize);

	CImg<double> imA, imB, imC, imZ(1, params.patchSize, 1, 1);
	imA.assign(grp.data, params.patchSize, params.patchSize, 1, 1);
	imB.assign(signal, params.patchSize, params.patchSize, 1, 1);
	//imC.assign(residual, BLOCK_SIZE, BLOCK_SIZE, 1, 1);
	imZ.fill(0);
	imA.append(imZ);
	imA.append(imB);
	//imA.append(imZ);
	//imA.append(imC);
	imA.display();

	mkl_free(list);
	mkl_free(L);
	mkl_free(L2);
	mkl_free(alpha);
	mkl_free(talpha);
	mkl_free(alphaTotal);
	mkl_free(residual);
	mkl_free(signal);
	mkl_free(tdict);
	mkl_free(corr);

	return;
}

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

