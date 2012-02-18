// cvLabelingImageLab: an impressively fast labeling routine for OpenCV
// Copyright (C) 2009 - Costantino Grana and Daniele Borghesani
//
// This library is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free 
// Software Foundation; either version 3 of the License, or (at your option) 
// any later version.
//
// This library is distributed in the hope that it will be useful, but WITHOUT 
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more 
// details.
//
// You should have received a copy of the GNU Lesser General Public License along
// with this library; if not, see <http://www.gnu.org/licenses/>.
//
// For further information contact us at
// Costantino Grana/Daniele Borghesani - University of Modena and Reggio Emilia - 
// Via Vignolese 905/b - 41100 Modena, Italy - e-mail: {name.surname}@unimore.it

#include "cvlabeling_imagelab.h"
#include "cxmisc.h"

static CvStatus icvLabelImage (IplImage* srcImage, IplImage* dstImage, unsigned char byF, int *numLabels);

// fast block based labeling with decision tree optimization
//
// src: single channel binary image of type IPL_DEPTH_8U 
// dst: single channel label image of type IPL_DEPTH_32S
CV_IMPL  void
cvLabelingImageLab (IplImage* srcImage, IplImage* dstImage, unsigned char byForeground, int *numLabels) {
	CV_FUNCNAME("cvLabelingImageLab");

    __BEGIN__;

	if (srcImage->nChannels!=1 || srcImage->depth!=IPL_DEPTH_8U)
		CV_ERROR( CV_StsUnsupportedFormat, "Input must be a single channel binary image of type IPL_DEPTH_8U" );
	if (dstImage->nChannels!=1 || dstImage->depth!=IPL_DEPTH_32S)
		CV_ERROR( CV_StsUnsupportedFormat, "Output must be a single channel label image of type IPL_DEPTH_32S" );
    if( srcImage->width!=dstImage->width || srcImage->height!=dstImage->height)
        CV_ERROR( CV_StsUnmatchedSizes, "The source and the destination images must be of the same size" );
    
    IPPI_CALL( icvLabelImage( srcImage, dstImage, byForeground, numLabels ));

    __END__;
}


static CvStatus icvLabelImage (IplImage* srcImage, IplImage* dstImage, unsigned char byF, int *numLabels) {
	int w(srcImage->width),h(srcImage->height),ws(srcImage->widthStep),wd(dstImage->widthStep);

	int iNewLabel(0);
	int *aRTable = new int[w*h/4];
	int *aNext = new int[w*h/4];
	int *aTail = new int[w*h/4];

	unsigned char *img = (unsigned char *)srcImage->imageData;
	char *imgOut = (char *)dstImage->imageData;

	for(int y=0; y<h; y+=2) {
		for(int x=0; x<w; x+=2) {

#define condition_a x-1>=0 && y-2>=0 && img[x-1+(y-2)*ws]==byF
#define condition_b y-2>=0 && img[x+(y-2)*ws]==byF
#define condition_c x+1<w && y-2>=0 && img[x+1+(y-2)*ws]==byF
#define condition_d x+2<w && y-2>=0 && img[x+2+(y-2)*ws]==byF
#define condition_e x-2>=0 && y-1>=0 && img[x-2+(y-1)*ws]==byF
#define condition_f x-1>=0 && y-1>=0 && img[x-1+(y-1)*ws]==byF
#define condition_g y-1>=0 && img[x+(y-1)*ws]==byF
#define condition_h x+1<w && y-1>=0 && img[x+1+(y-1)*ws]==byF
#define condition_i x+2<w && y-1>=0 && img[x+2+(y-1)*ws]==byF
#define condition_j x-2>=0 && img[x-2+(y)*ws]==byF
#define condition_k x-1>=0 && img[x-1+(y)*ws]==byF
#define condition_l img[x+(y)*ws]==byF
#define condition_m x+1<w && img[x+1+(y)*ws]==byF
#define condition_n x-1>=0 && y+1<h && img[x-1+(y+1)*ws]==byF
#define condition_o y+1<h && img[x+(y+1)*ws]==byF
#define condition_p x+1<w && y+1<h && img[x+1+(y+1)*ws]==byF

			if (condition_l) {
				if (condition_k) {
					if (condition_i) {
						if (condition_h) {
							if (condition_g) {
								goto action_4;
							}
							else {
								if (condition_b) {
									if (condition_f) {
										goto action_4;
									}
									else {
										if (condition_e) {
											if (condition_a) {
												goto action_4;
											}
											else {
												goto action_10;
											}
										}
										else {
											goto action_10;
										}
									}
								}
								else {
									goto action_10;
								}
							}
						}
						else {
							if (condition_m) {
								if (condition_c) {
									if (condition_g) {
										goto action_4;
									}
									else {
										if (condition_b) {
											if (condition_f) {
												goto action_4;
											}
											else {
												if (condition_e) {
													if (condition_a) {
														goto action_4;
													}
													else {
														goto action_10;
													}
												}
												else {
													goto action_10;
												}
											}
										}
										else {
											goto action_10;
										}
									}
								}
								else {
									goto action_10;
								}
							}
							else {
								if (condition_g) {
									goto action_4;
								}
								else {
									if (condition_b) {
										if (condition_f) {
											goto action_4;
										}
										else {
											if (condition_e) {
												if (condition_a) {
													goto action_4;
												}
												else {
													goto action_6;
												}
											}
											else {
												goto action_6;
											}
										}
									}
									else {
										goto action_6;
									}
								}
							}
						}
					}
					else {
						if (condition_g) {
							goto action_4;
						}
						else {
							if (condition_b) {
								if (condition_f) {
									goto action_4;
								}
								else {
									if (condition_e) {
										if (condition_a) {
											goto action_4;
										}
										else {
											if (condition_h) {
												if (condition_d) {
													goto action_10;
												}
												else {
													goto action_11;
												}
											}
											else {
												goto action_6;
											}
										}
									}
									else {
										if (condition_h) {
											if (condition_d) {
												goto action_10;
											}
											else {
												goto action_11;
											}
										}
										else {
											goto action_6;
										}
									}
								}
							}
							else {
								if (condition_h) {
									if (condition_d) {
										goto action_10;
									}
									else {
										goto action_11;
									}
								}
								else {
									goto action_6;
								}
							}
						}
					}
				}
				else {
					if (condition_h) {
						if (condition_f) {
							if (condition_j) {
								if (condition_g) {
									goto action_4;
								}
								else {
									if (condition_b) {
										goto action_4;
									}
									else {
										if (condition_i) {
											goto action_10;
										}
										else {
											if (condition_d) {
												goto action_10;
											}
											else {
												goto action_11;
											}
										}
									}
								}
							}
							else {
								if (condition_n) {
									if (condition_i) {
										if (condition_g) {
											goto action_10;
										}
										else {
											if (condition_b) {
												goto action_10;
											}
											else {
												goto action_14;
											}
										}
									}
									else {
										if (condition_g) {
											if (condition_d) {
												goto action_10;
											}
											else {
												goto action_11;
											}
										}
										else {
											if (condition_d) {
												if (condition_b) {
													goto action_10;
												}
												else {
													goto action_14;
												}
											}
											else {
												if (condition_b) {
													goto action_11;
												}
												else {
													goto action_15;
												}
											}
										}
									}
								}
								else {
									if (condition_g) {
										goto action_4;
									}
									else {
										if (condition_b) {
											goto action_4;
										}
										else {
											if (condition_i) {
												goto action_8;
											}
											else {
												if (condition_d) {
													goto action_8;
												}
												else {
													goto action_9;
												}
											}
										}
									}
								}
							}
						}
						else {
							if (condition_n) {
								if (condition_j) {
									if (condition_e) {
										if (condition_a) {
											if (condition_g) {
												goto action_4;
											}
											else {
												if (condition_b) {
													goto action_4;
												}
												else {
													if (condition_i) {
														goto action_10;
													}
													else {
														if (condition_d) {
															goto action_10;
														}
														else {
															goto action_11;
														}
													}
												}
											}
										}
										else {
											if (condition_i) {
												goto action_10;
											}
											else {
												if (condition_d) {
													goto action_10;
												}
												else {
													goto action_11;
												}
											}
										}
									}
									else {
										if (condition_i) {
											goto action_10;
										}
										else {
											if (condition_d) {
												goto action_10;
											}
											else {
												goto action_11;
											}
										}
									}
								}
								else {
									if (condition_i) {
										goto action_10;
									}
									else {
										if (condition_d) {
											goto action_10;
										}
										else {
											goto action_11;
										}
									}
								}
							}
							else {
								goto action_4;
							}
						}
					}
					else {
						if (condition_i) {
							if (condition_m) {
								if (condition_f) {
									if (condition_j) {
										if (condition_c) {
											if (condition_g) {
												goto action_4;
											}
											else {
												if (condition_b) {
													goto action_4;
												}
												else {
													goto action_10;
												}
											}
										}
										else {
											goto action_10;
										}
									}
									else {
										if (condition_n) {
											if (condition_c) {
												if (condition_g) {
													goto action_10;
												}
												else {
													if (condition_b) {
														goto action_10;
													}
													else {
														goto action_14;
													}
												}
											}
											else {
												goto action_14;
											}
										}
										else {
											if (condition_c) {
												if (condition_g) {
													goto action_4;
												}
												else {
													if (condition_b) {
														goto action_4;
													}
													else {
														goto action_8;
													}
												}
											}
											else {
												goto action_8;
											}
										}
									}
								}
								else {
									if (condition_c) {
										if (condition_n) {
											if (condition_j) {
												if (condition_e) {
													if (condition_a) {
														if (condition_g) {
															goto action_4;
														}
														else {
															if (condition_b) {
																goto action_4;
															}
															else {
																goto action_10;
															}
														}
													}
													else {
														goto action_10;
													}
												}
												else {
													goto action_10;
												}
											}
											else {
												goto action_10;
											}
										}
										else {
											goto action_4;
										}
									}
									else {
										if (condition_g) {
											if (condition_a) {
												if (condition_j) {
													if (condition_e) {
														goto action_10;
													}
													else {
														if (condition_n) {
															goto action_14;
														}
														else {
															goto action_8;
														}
													}
												}
												else {
													if (condition_n) {
														goto action_14;
													}
													else {
														goto action_8;
													}
												}
											}
											else {
												if (condition_n) {
													goto action_13;
												}
												else {
													goto action_7;
												}
											}
										}
										else {
											if (condition_n) {
												goto action_10;
											}
											else {
												goto action_3;
											}
										}
									}
								}
							}
							else {
								if (condition_n) {
									if (condition_j) {
										if (condition_g) {
											if (condition_f) {
												goto action_4;
											}
											else {
												if (condition_e) {
													if (condition_a) {
														goto action_4;
													}
													else {
														if (condition_c) {
															goto action_10;
														}
														else {
															goto action_11;
														}
													}
												}
												else {
													if (condition_c) {
														goto action_10;
													}
													else {
														goto action_11;
													}
												}
											}
										}
										else {
											if (condition_b) {
												if (condition_f) {
													goto action_4;
												}
												else {
													if (condition_e) {
														if (condition_a) {
															goto action_4;
														}
														else {
															goto action_6;
														}
													}
													else {
														goto action_6;
													}
												}
											}
											else {
												goto action_6;
											}
										}
									}
									else {
										if (condition_g) {
											if (condition_c) {
												goto action_10;
											}
											else {
												goto action_11;
											}
										}
										else {
											if (condition_f) {
												if (condition_b) {
													if (condition_c) {
														goto action_10;
													}
													else {
														goto action_11;
													}
												}
												else {
													goto action_12;
												}
											}
											else {
												goto action_6;
											}
										}
									}
								}
								else {
									if (condition_g) {
										goto action_4;
									}
									else {
										if (condition_f) {
											if (condition_b) {
												goto action_4;
											}
											else {
												if (condition_j) {
													goto action_6;
												}
												else {
													goto action_5;
												}
											}
										}
										else {
											goto action_2;
										}
									}
								}
							}
						}
						else {
							if (condition_n) {
								if (condition_j) {
									if (condition_g) {
										if (condition_f) {
											goto action_4;
										}
										else {
											if (condition_e) {
												if (condition_a) {
													goto action_4;
												}
												else {
													if (condition_d) {
														if (condition_c) {
															goto action_10;
														}
														else {
															goto action_11;
														}
													}
													else {
														goto action_11;
													}
												}
											}
											else {
												if (condition_d) {
													if (condition_c) {
														goto action_10;
													}
													else {
														goto action_11;
													}
												}
												else {
													goto action_11;
												}
											}
										}
									}
									else {
										if (condition_b) {
											if (condition_f) {
												goto action_4;
											}
											else {
												if (condition_e) {
													if (condition_a) {
														goto action_4;
													}
													else {
														goto action_6;
													}
												}
												else {
													goto action_6;
												}
											}
										}
										else {
											goto action_6;
										}
									}
								}
								else {
									if (condition_g) {
										if (condition_d) {
											if (condition_c) {
												goto action_10;
											}
											else {
												goto action_11;
											}
										}
										else {
											goto action_11;
										}
									}
									else {
										if (condition_f) {
											if (condition_b) {
												if (condition_d) {
													if (condition_c) {
														goto action_10;
													}
													else {
														goto action_11;
													}
												}
												else {
													goto action_11;
												}
											}
											else {
												goto action_12;
											}
										}
										else {
											goto action_6;
										}
									}
								}
							}
							else {
								if (condition_g) {
									goto action_4;
								}
								else {
									if (condition_f) {
										if (condition_b) {
											goto action_4;
										}
										else {
											if (condition_j) {
												goto action_6;
											}
											else {
												goto action_5;
											}
										}
									}
									else {
										goto action_2;
									}
								}
							}
						}
					}
				}
			}
			else {
				if (condition_m) {
					if (condition_h) {
						if (condition_o) {
							if (condition_k) {
								if (condition_g) {
									goto action_4;
								}
								else {
									if (condition_b) {
										if (condition_f) {
											goto action_4;
										}
										else {
											if (condition_e) {
												if (condition_a) {
													goto action_4;
												}
												else {
													if (condition_i) {
														goto action_10;
													}
													else {
														if (condition_d) {
															goto action_10;
														}
														else {
															goto action_11;
														}
													}
												}
											}
											else {
												if (condition_i) {
													goto action_10;
												}
												else {
													if (condition_d) {
														goto action_10;
													}
													else {
														goto action_11;
													}
												}
											}
										}
									}
									else {
										if (condition_i) {
											goto action_10;
										}
										else {
											if (condition_d) {
												goto action_10;
											}
											else {
												goto action_11;
											}
										}
									}
								}
							}
							else {
								if (condition_n) {
									if (condition_j) {
										if (condition_f) {
											if (condition_g) {
												goto action_4;
											}
											else {
												if (condition_b) {
													goto action_4;
												}
												else {
													if (condition_i) {
														goto action_10;
													}
													else {
														if (condition_d) {
															goto action_10;
														}
														else {
															goto action_11;
														}
													}
												}
											}
										}
										else {
											if (condition_e) {
												if (condition_a) {
													if (condition_g) {
														goto action_4;
													}
													else {
														if (condition_b) {
															goto action_4;
														}
														else {
															if (condition_i) {
																goto action_10;
															}
															else {
																if (condition_d) {
																	goto action_10;
																}
																else {
																	goto action_11;
																}
															}
														}
													}
												}
												else {
													if (condition_i) {
														goto action_10;
													}
													else {
														if (condition_d) {
															goto action_10;
														}
														else {
															goto action_11;
														}
													}
												}
											}
											else {
												if (condition_i) {
													goto action_10;
												}
												else {
													if (condition_d) {
														goto action_10;
													}
													else {
														goto action_11;
													}
												}
											}
										}
									}
									else {
										if (condition_i) {
											goto action_10;
										}
										else {
											if (condition_d) {
												goto action_10;
											}
											else {
												goto action_11;
											}
										}
									}
								}
								else {
									goto action_4;
								}
							}
						}
						else {
							goto action_4;
						}
					}
					else {
						if (condition_i) {
							if (condition_c) {
								if (condition_o) {
									if (condition_k) {
										if (condition_g) {
											goto action_4;
										}
										else {
											if (condition_b) {
												if (condition_f) {
													goto action_4;
												}
												else {
													if (condition_e) {
														if (condition_a) {
															goto action_4;
														}
														else {
															goto action_10;
														}
													}
													else {
														goto action_10;
													}
												}
											}
											else {
												goto action_10;
											}
										}
									}
									else {
										if (condition_n) {
											if (condition_j) {
												if (condition_f) {
													if (condition_g) {
														goto action_4;
													}
													else {
														if (condition_b) {
															goto action_4;
														}
														else {
															goto action_10;
														}
													}
												}
												else {
													if (condition_e) {
														if (condition_a) {
															if (condition_g) {
																goto action_4;
															}
															else {
																if (condition_b) {
																	goto action_4;
																}
																else {
																	goto action_10;
																}
															}
														}
														else {
															goto action_10;
														}
													}
													else {
														goto action_10;
													}
												}
											}
											else {
												goto action_10;
											}
										}
										else {
											goto action_4;
										}
									}
								}
								else {
									goto action_4;
								}
							}
							else {
								if (condition_g) {
									if (condition_k) {
										goto action_10;
									}
									else {
										if (condition_f) {
											if (condition_j) {
												goto action_10;
											}
											else {
												if (condition_o) {
													if (condition_n) {
														goto action_14;
													}
													else {
														goto action_8;
													}
												}
												else {
													goto action_8;
												}
											}
										}
										else {
											if (condition_a) {
												if (condition_j) {
													if (condition_e) {
														goto action_10;
													}
													else {
														if (condition_o) {
															if (condition_n) {
																goto action_14;
															}
															else {
																goto action_8;
															}
														}
														else {
															goto action_8;
														}
													}
												}
												else {
													if (condition_o) {
														if (condition_n) {
															goto action_14;
														}
														else {
															goto action_8;
														}
													}
													else {
														goto action_8;
													}
												}
											}
											else {
												if (condition_o) {
													if (condition_n) {
														goto action_13;
													}
													else {
														goto action_7;
													}
												}
												else {
													goto action_7;
												}
											}
										}
									}
								}
								else {
									if (condition_o) {
										if (condition_n) {
											goto action_10;
										}
										else {
											if (condition_k) {
												goto action_10;
											}
											else {
												goto action_3;
											}
										}
									}
									else {
										goto action_3;
									}
								}
							}
						}
						else {
							if (condition_o) {
								if (condition_k) {
									if (condition_g) {
										goto action_4;
									}
									else {
										if (condition_b) {
											if (condition_f) {
												goto action_4;
											}
											else {
												if (condition_e) {
													if (condition_a) {
														goto action_4;
													}
													else {
														goto action_6;
													}
												}
												else {
													goto action_6;
												}
											}
										}
										else {
											goto action_6;
										}
									}
								}
								else {
									if (condition_n) {
										if (condition_j) {
											if (condition_g) {
												if (condition_f) {
													goto action_4;
												}
												else {
													if (condition_e) {
														if (condition_a) {
															goto action_4;
														}
														else {
															if (condition_d) {
																if (condition_c) {
																	goto action_10;
																}
																else {
																	goto action_11;
																}
															}
															else {
																goto action_11;
															}
														}
													}
													else {
														if (condition_d) {
															if (condition_c) {
																goto action_10;
															}
															else {
																goto action_11;
															}
														}
														else {
															goto action_11;
														}
													}
												}
											}
											else {
												if (condition_b) {
													if (condition_f) {
														goto action_4;
													}
													else {
														if (condition_e) {
															if (condition_a) {
																goto action_4;
															}
															else {
																goto action_6;
															}
														}
														else {
															goto action_6;
														}
													}
												}
												else {
													goto action_6;
												}
											}
										}
										else {
											if (condition_g) {
												if (condition_d) {
													if (condition_c) {
														goto action_10;
													}
													else {
														goto action_11;
													}
												}
												else {
													goto action_11;
												}
											}
											else {
												goto action_6;
											}
										}
									}
									else {
										if (condition_g) {
											goto action_4;
										}
										else {
											goto action_2;
										}
									}
								}
							}
							else {
								if (condition_g) {
									goto action_4;
								}
								else {
									goto action_2;
								}
							}
						}
					}
				}
				else {
					if (condition_o) {
						if (condition_k) {
							if (condition_g) {
								goto action_4;
							}
							else {
								if (condition_b) {
									if (condition_f) {
										goto action_4;
									}
									else {
										if (condition_e) {
											if (condition_a) {
												goto action_4;
											}
											else {
												goto action_6;
											}
										}
										else {
											goto action_6;
										}
									}
								}
								else {
									goto action_6;
								}
							}
						}
						else {
							if (condition_n) {
								if (condition_j) {
									if (condition_f) {
										if (condition_g) {
											goto action_4;
										}
										else {
											if (condition_b) {
												goto action_4;
											}
											else {
												goto action_6;
											}
										}
									}
									else {
										if (condition_e) {
											if (condition_a) {
												if (condition_g) {
													goto action_4;
												}
												else {
													if (condition_b) {
														goto action_4;
													}
													else {
														goto action_6;
													}
												}
											}
											else {
												goto action_6;
											}
										}
										else {
											goto action_6;
										}
									}
								}
								else {
									goto action_6;
								}
							}
							else {
								goto action_2;
							}
						}
					}
					else {
						if (condition_p) {
							goto action_2;
						}
						else {
							goto action_1;
						}
					}
				}
			}

			int lx,u,v,k;

#define INT_PTR(x) (*((int*)(&(x))))

action_1:	lx = 0;
			goto fine;
action_2:	lx = ++iNewLabel;
			aRTable[lx] = lx;
			aNext[lx] = -1;
			aTail[lx] = lx;
			goto fine;
action_3:	lx = INT_PTR(imgOut[(x+2)*4+(y-2)*wd]);
			goto fine;
action_4:	lx = INT_PTR(imgOut[(x)*4+(y-2)*wd]);
			goto fine;
action_5:	lx = INT_PTR(imgOut[(x-2)*4+(y-2)*wd]);
			goto fine;
action_6:	lx = INT_PTR(imgOut[(x-2)*4+(y)*wd]);
			goto fine;
action_7:	lx = INT_PTR(imgOut[(x)*4+(y-2)*wd]);
			u = aRTable[lx];
			v = aRTable[INT_PTR(imgOut[(x+2)*4+(y-2)*wd])];
			goto merge2;
action_8:	lx = INT_PTR(imgOut[(x-2)*4+(y-2)*wd]);
			u = aRTable[lx];
			v = aRTable[INT_PTR(imgOut[(x+2)*4+(y-2)*wd])];
			goto merge2;
action_9:	lx = INT_PTR(imgOut[(x-2)*4+(y-2)*wd]);
			u = aRTable[lx];
			v = aRTable[INT_PTR(imgOut[(x)*4+(y-2)*wd])];
			goto merge2;
action_10:	lx = INT_PTR(imgOut[(x-2)*4+(y)*wd]);
			u = aRTable[lx];
			v = aRTable[INT_PTR(imgOut[(x+2)*4+(y-2)*wd])];
			goto merge2;
action_11:	lx = INT_PTR(imgOut[(x-2)*4+(y)*wd]);
			u = aRTable[lx];
			v = aRTable[INT_PTR(imgOut[(x)*4+(y-2)*wd])];
			goto merge2;
action_12:	lx = INT_PTR(imgOut[(x-2)*4+(y)*wd]);
			u = aRTable[lx];
			v = aRTable[INT_PTR(imgOut[(x-2)*4+(y-2)*wd])];
			goto merge2;
action_13:	lx = INT_PTR(imgOut[(x-2)*4+(y)*wd]);
			u = aRTable[lx];
			v = aRTable[INT_PTR(imgOut[(x)*4+(y-2)*wd])];
			//k = aRTable[imgOut[(x+2)*4+(y-2)*wd]];
			k = INT_PTR(imgOut[(x+2)*4+(y-2)*wd]);
			goto merge3;
action_14:	lx = INT_PTR(imgOut[(x-2)*4+(y)*wd]);
			u = aRTable[lx];
			v = aRTable[INT_PTR(imgOut[(x-2)*4+(y-2)*wd])];
			//k = aRTable[imgOut[(x+2)*4+(y-2)*wd]];
			k = INT_PTR(imgOut[(x+2)*4+(y-2)*wd]);
			goto merge3;
action_15:	lx = INT_PTR(imgOut[(x-2)*4+(y)*wd]);
			u = aRTable[lx];
			v = aRTable[INT_PTR(imgOut[(x-2)*4+(y-2)*wd])];
			//k = aRTable[imgOut[(x)*4+(y-2)*wd]];
			k = INT_PTR(imgOut[(x)*4+(y-2)*wd]);
			//goto merge3;

merge3:		if (u<v) {
				int i = v;
				while (i>-1) {
					aRTable[i] = u;
					i = aNext[i];
				}
				aNext[aTail[u]] = v;
				aTail[u] = aTail[v];

				k = aRTable[k];
				if (u<k) {
					int i = k;
					while (i>-1) {
						aRTable[i] = u;
						i = aNext[i];
					}
					aNext[aTail[u]] = k;
					aTail[u] = aTail[k];
				}
				else if (u>k) {
					int i = u;
					while (i>-1) {
						aRTable[i] = k;
						i = aNext[i];
					}
					aNext[aTail[k]] = u;
					aTail[k] = aTail[u];
				}
			}
			else if (u>v) {
				int i = u;
				while (i>-1) {
					aRTable[i] = v;
					i = aNext[i];
				}
				aNext[aTail[v]] = u;
				aTail[v] = aTail[u];

				k = aRTable[k];
				if (v<k) {
					int i = k;
					while (i>-1) {
						aRTable[i] = v;
						i = aNext[i];
					}
					aNext[aTail[v]] = k;
					aTail[v] = aTail[k];
				}
				else if (v>k) {
					int i = v;
					while (i>-1) {
						aRTable[i] = k;
						i = aNext[i];
					}
					aNext[aTail[k]] = v;
					aTail[k] = aTail[v];
				}
			}
			else {
				k = aRTable[k];
				if (u<k) {
					int i = k;
					while (i>-1) {
						aRTable[i] = u;
						i = aNext[i];
					}
					aNext[aTail[u]] = k;
					aTail[u] = aTail[k];
				}
				else if (u>k) {
					int i = u;
					while (i>-1) {
						aRTable[i] = k;
						i = aNext[i];
					}
					aNext[aTail[k]] = u;
					aTail[k] = aTail[u];
				}
			}

			goto fine;

merge2:		if (u<v) {
				 int i = v;
				 while (i>-1) {
					 aRTable[i] = u;
					 i = aNext[i];
				 }
				 aNext[aTail[u]] = v;
				 aTail[u] = aTail[v];
			}
			else if (u>v) {
				 int i = u;
				 while (i>-1) {
					 aRTable[i] = v;
					 i = aNext[i];
				 }
				 aNext[aTail[v]] = u;
				 aTail[v] = aTail[u];
			}
			//goto fine;
//fine:		memset(imgOut+x*4+y*wd,lx,sizeof(int));
fine:		INT_PTR(imgOut[x*4+y*wd]) = lx;

		}
	}

	// Rinumero le label
	int iCurLabel = 0;
	for (int k=1;k<=iNewLabel;k++) {
		if (aRTable[k]==k) {
			iCurLabel++;
			aRTable[k] = iCurLabel;
		}
		else
			aRTable[k] = aRTable[aRTable[k]];
	}

	// SECOND SCAN 
	for(int y=0;y<h;y+=2) {
		for(int x=0;x<w;x+=2) {
			int iLabel = INT_PTR(imgOut[x*4+y*wd]) ;
			if (iLabel>0) {
				iLabel = aRTable[iLabel];
				if (img[x+y*ws]==byF)
					INT_PTR(imgOut[x*4+y*wd]) = iLabel;
				else
					INT_PTR(imgOut[x*4+y*wd]) = 0;
				if (x+1<w) {
					if (img[x+1+y*ws]==byF)
						INT_PTR(imgOut[(x+1)*4+y*wd]) = iLabel;
					else
						INT_PTR(imgOut[(x+1)*4+y*wd]) = 0;
					if (y+1<h) {
						if (img[x+(y+1)*ws]==byF)
							INT_PTR(imgOut[(x)*4+(y+1)*wd]) = iLabel;
						else
							INT_PTR(imgOut[(x)*4+(y+1)*wd]) = 0;
						if (img[x+1+(y+1)*ws]==byF)
							INT_PTR(imgOut[(x+1)*4+(y+1)*wd]) = iLabel;
						else
							INT_PTR(imgOut[(x+1)*4+(y+1)*wd]) = 0;
					}
				}
				else if (y+1<h) {
					if (img[x+(y+1)*ws]==byF)
						INT_PTR(imgOut[(x)*4+(y+1)*wd]) = iLabel;
					else
						INT_PTR(imgOut[(x)*4+(y+1)*wd]) = 0;
				}
			}
			else {
				INT_PTR(imgOut[(x)*4+(y)*wd]) = 0;
				if (x+1<w) {
					INT_PTR(imgOut[(x+1)*4+y*wd]) = 0;
					if (y+1<h) {
						INT_PTR(imgOut[(x)*4+(y+1)*wd]) = 0;
						INT_PTR(imgOut[(x+1)*4+(y+1)*wd]) = 0;
					}
				}
				else if (y+1<h) {
					INT_PTR(imgOut[(x)*4+(y+1)*wd]) = 0;
				}
			}
		}
	}

	// output the number of labels
	*numLabels = iCurLabel;

	delete aRTable;
	delete aNext;
	delete aTail;

	return CV_OK;
}
