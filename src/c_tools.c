#include "c_tools.h"

///////////////////////////////////////////////////////////////////////////////
//  Function that allocates space and creates a double matrix.
//  Input: Dimension of the matrix to be created//  Output: Pointer to the created matrix.
///////////////////////////////////////////////////////////////////////////////
double **create_matrix(int rows, int columns)
{
    double **a;
    int i=0;
    a = (double**) calloc(rows, sizeof(double*));
    for(i=0;i<rows;i++) a[i] = (double*) calloc(columns,sizeof(double));
    return a;
}

///////////////////////////////////////////////////////////////////////////////
//  Function that frees the space that a double matrix has been allocated.
//  Input: Dimension of the matrix and a pointer to the matrix.
//  Output: Void.
///////////////////////////////////////////////////////////////////////////////
void free_matrix(double **a, int rows)
{
    int i=0;
    for(i=0;i<rows;i++) free(a[i]);
    free(a);
}

////////////////////////////////////////////////////////////////////////
// Fast computation of Kendall's tau
//
// Given by Shing (Eric) Fu, Feng Zhu, Guang (Jack) Yang, and Harry Joe
// Based on work of the method by Knight (1966)
/////////////////////////////////////////////////////////////////////////

void ktau(double *X, double *Y, int *N, double *tau, double *S, double *D, int *T, int *U, int *V)
{
	// Defining variables
	int K, L, I, J, Iend, Jend;
	int i, j, m;
	double *Y2 = calloc(*N, sizeof(double));
	double *X2 = calloc(*N, sizeof(double));
	double *xptr,*yptr; // HJ addition for swapping
	bool Iflag, Jflag, Xflag;
	*S = 0.; *D = 0.; *T = 0; *U = 0; *V = 0;

	/* 1.1 Sort X and Y in X order */
	/* Break ties in X according to Y */
	K=1;
	do
	{
		L=0;
		do
		{
			I = L;
			J = (I+K)<(*N)?(I+K):(*N);
			Iend = J;
			Jend = (J+K)<(*N)?(J+K):(*N);
			do
			{
				Iflag = (I < Iend);
				Jflag = (J < Jend);
				if (Iflag & Jflag)
				{
				 	Xflag = ((X[I] > X[J]) | ((X[I] == X[J]) & (Y[I] > Y[J])));
				}
				else
				{
					Xflag = false;
				}
				if((Iflag & !Jflag) | (Iflag & Jflag & !Xflag))
				{
					X2[L] = X[I];
					Y2[L] = Y[I];
					I++;
					L++;
				};
				if((!Iflag && Jflag) | (Iflag && Jflag && Xflag))
				{
					X2[L] = X[J];
					Y2[L] = Y[J];
					J++;
					L++;
				};
			}
			while(Iflag | Jflag);
		}
		while(L < *N);

		// Swap lists
		xptr=X; X=X2; X2=xptr;
		yptr=Y; Y=Y2; Y2=yptr;
		#ifdef OLD
		for(i = 0; i < *N; i++)
		{
			Xtem = X[i]; Ytem = Y[i];
			X[i] = X2[i]; Y[i] = Y2[i];
			X2[i] = Xtem; Y2[i] = Ytem;
		};
		#endif
		K *= 2;
	}
	while (K < *N);

	/* 1.2 Count pairs of tied X, T */
	j = 1;
	m = 1;
	for(i = 1; i < *N; i++)
    if(X[i] == X[i-1])
    {
		j++;
		if(Y[i] == Y[i-1])
		m++;
    }
    else if(j > 1)
    {
      *T += j * (j - 1) / 2;
      if(m > 1)
		*V += m * (m - 1) / 2;
      j = 1;
      m = 1;
    };
	*T += j * (j - 1) / 2;
	*V += m * (m - 1) / 2;

	/* 2.1 Sort Y again and count exchanges, S */
	/* Keep original relative order if tied */

	K=1;
	do
	{
		L=0;
		do
		{
			I = L;
			J = (I+K)<(*N)?(I+K):(*N);
			Iend = J;
			Jend = (J+K)<(*N)?(J+K):(*N);
			do
			{
				Iflag = (I < Iend);
				Jflag = (J < Jend);
				if (Iflag & Jflag)
				{
				 	Xflag = (Y[I] > Y[J]);
				}
				else
				{
					Xflag = false;
				}
				if((Iflag & !Jflag) | (Iflag & Jflag & !Xflag))
				{
					X2[L] = X[I];
					Y2[L] = Y[I];
					I++;
					L++;
				};
				if((!Iflag && Jflag) | (Iflag && Jflag && Xflag))
				{
					X2[L] = X[J];
					Y2[L] = Y[J];
					*S += Iend - I;
					J++;
					L++;
				};
			}
			while((Iflag | Jflag));
		}
		while(L < *N);

		// Swap lists
		xptr=X; X=X2; X2=xptr;
		yptr=Y; Y=Y2; Y2=yptr;
		#ifdef OLD
		for(i = 0; i < *N; i++)
		{
			Xtem = X[i]; Ytem = Y[i];
			X[i] = X2[i]; Y[i] = Y2[i];
			X2[i] = Xtem; Y2[i] = Ytem;
		};
		#endif
		K *= 2;
	}
	while (K < *N);

	/* 2.2 Count pairs of tied Y, U */
	j=1;
	for(i = 1; i < *N; i++)
		if(Y[i] == Y[i-1])
			j++;
		else if(j > 1)
		{
			*U += j * (j - 1) / 2;
			j = 1;
		};
	*U += j * (j - 1) / 2;


	/* 3. Calc. Kendall's Score and Denominator */
	*D = 0.5 * (*N) * (*N - 1);
	*S = *D - (2. * (*S) + *T + *U - *V);
	//if(*T > 0 | *U > 0) // adjust for ties
    *D = sqrt((*D - *T) * (*D - *U));
	*tau = (*S) / (*D);


  free(Y2);
  free(X2);
}


//////////////////////////////////////////////////
// ktau_matrix
//
// Input:
// data			data vector
// d			data dimension 1
// N			data dimension 2
//
// Output:
// out			Kendall's tau Matrix (as vector)

void ktau_matrix(double *data, int *d, int *N, double *out)
{
	double **x, S=0.0, D=0.0, *X, *Y;
	int k=0, i, j, t, T=0, U=0, V=0;
	x = create_matrix(*d,*N);
	X = (double*) calloc(*N,sizeof(double));
	Y = (double*) calloc(*N,sizeof(double));

	for(i=0;i<*d;i++)
    {
		for (t=0;t<*N;t++ )
		{
			x[i][t] = data[k];
			k++;
		}
    }

	k=0;
	for(i=0;i<((*d)-1);i++)
	{
		for(j=(i+1);j<(*d);j++)
		{
			for(t=0;t<*N;t++)
			{
				X[t]=x[i][t];
				Y[t]=x[j][t];
			}
			ktau(X, Y, N, &out[k], &S, &D, &T, &U, &V);
			k++;
		}
	}

	free(X);free(Y);free_matrix(x, *d);
}

/** 1/(2pi) */
#define M_1_2PI .159154943091895335768883763373

/** Debye function of order n.
    int(t=0..x) t^n dt / [exp(t)-1]
 The underivative for n=0 is log[1-exp(-x)], which is infinite at x=0,
 so the corresponding Debye-function is not defined at n=0.

 Literature:
 Ng et al, Math. Comp. 24 (110) (1970) 405
 Guseinov et al, Intl. J. Thermophys. 28 (4) (2007) 1420
 Engeln et al, Colloid & Polymer Sci. 261 (9) (1983) 736
 Maximon , Proc. R. Soc. A 459 (2039) (2003) 2807

 @param[in] x the argument and upper limit of the integral. x>=0.
 @param[in] n the power in the numerator of the integral, 1<=n<=20 .
 @return the Debye function. Zero if x<=0, and -1 if n is outside the
     parameter range that is implemented.
 @author Richard J. Mathar
 @since 2007-10-31 implemented range n=8..10
*/
double debyen(const double x, const int n)
{
    if (x<=0. )
        return 0. ;
    if ( n <1 || n >20)
        return -1. ;

    /* for values up to 4.80 the list of zeta functions
    and the sum up to k < K are huge enough to gain
    numeric stability in the sum */
    if(x>= 3. )
    {
        double sum ;
        /* list of n! zeta(n+1) for n =0 up to the maximum n implemented.
        Limited by the cancellation of digits encountered at smaller x and larger n.
        Digits := 30 :
        for n from 1 to 30 do
                printf("%.29e, ", evalf(n!*Zeta(n+1))) ;
        od:
        */
        static double nzetan[] = { 0., 1.64493406684822643647241516665e+00, 2.40411380631918857079947632302e+00,
                                   6.49393940226682914909602217926e+00, 2.48862661234408782319527716750e+01,
                                   1.22081167438133896765742151575e+02, 7.26011479714984435324654235892e+02,
                                   5.06054987523763947046857360209e+03, 4.04009783987476348853278236554e+04,
                                   3.63240911422382626807143525567e+05, 3.63059331160662871299061884284e+06,
                                   3.99266229877310867023270732405e+07, 4.79060379889831452426876764501e+08,
                                   6.22740219341097176419285340896e+09, 8.71809578301720678451912203103e+10,
                                   1.30769435221891382089009990749e+12, 2.09229496794815109066316556880e+13,
                                   3.55688785859223715975612396717e+14, 6.40238592281892140073564945334e+15,
                                   1.21645216453639396669876696274e+17, 2.43290316850786132173725681824e+18,
                                   5.10909543543702856776502748606e+19, 1.12400086178089123060215294900e+21
        } ;

        /* constrained to the list of nzetan[] given above */
        if ( n >= sizeof(nzetan)/sizeof(double) )
            return -1. ;

        /* n!*zeta(n) is the integral for x=infinity , 27.1.3 */
        sum = nzetan[n] ;

        /* the number of terms needed in the k-sum for x=0,1,2,3...
        * Reflects the n=1 case, because higher n need less terms.
        */
        static int kLim[] = {0,0,0,13,10,8,7,6,5,5,4,4,4,3} ;

        const int kmax = ((int) x < sizeof(kLim)/sizeof(int)) ? kLim[(int)x] : 3 ;
        /* Abramowitz Stegun 27.1.2 */
        int k;
        for(k=1; k<=kmax ;k++)
        {
            /* do not use x(k+1)=xk+x to avoid loss of precision */
            const double xk = x*k ;
            double ksum= 1./xk ;
            double tmp = n*ksum/xk ;	/* n/(xk)^2 */
            int s;
            for (s=1 ; s <= n ; s++)
            {
                ksum += tmp ;
                tmp *= (n-s)/xk ;
            }
            sum -= exp(-xk)* ksum*pow(x,n+1.) ;
        }
        return sum ;
    }
    else
    {
        /* list of absolute values of Bernoulli numbers of index 2*k, multiplied  by
        (2*pi)^k/(2k)!, and 2 subtracted, k=0,1,2,3,4
        Digits := 60 :
        interface(prettyprint=0) :
        for k from 1 to 70 do
         printf("%.30e,\n",evalf( abs((2*Pi)^(2*k)*bernoulli(2*k)/(2*k)!)-2 )) ;
        od;
        */
        static double koeff[]= { 0., 1.289868133696452872944830333292e+00, 1.646464674222763830320073930823e-01,
                                 3.468612396889827942903585958184e-02, 8.154712395888678757370477017305e-03,
                                 1.989150255636170674291917800638e-03, 4.921731066160965972759960954793e-04,
                                 1.224962701174096585170902102707e-04, 3.056451881730374346514297527344e-05,
                                 7.634586529999679712923289243879e-06, 1.907924067745592226304077366899e-06,
                                 4.769010054554659800072963735060e-07, 1.192163781025189592248804158716e-07,
                                 2.980310965673008246931701326140e-08, 7.450668049576914109638408036805e-09,
                                 1.862654864839336365743529470042e-09, 4.656623667353010984002911951881e-10,
                                 1.164154417580540177848737197821e-10, 2.910384378208396847185926449064e-11,
                                 7.275959094757302380474472711747e-12, 1.818989568052777856506623677390e-12,
                                 4.547473691649305030453643155957e-13, 1.136868397525517121855436593505e-13,
                                 2.842170965606321353966861428348e-14, 7.105427382674227346596939068119e-15,
                                 1.776356842186163180619218277278e-15, 4.440892101596083967998640188409e-16,
                                 1.110223024969096248744747318102e-16, 2.775557561945046552567818981300e-17,
                                 6.938893904331845249488542992219e-18, 1.734723476023986745668411013469e-18,
                                 4.336808689994439570027820336642e-19, 1.084202172491329082183740080878e-19,
                                 2.710505431220232916297046799365e-20, 6.776263578041593636171406200902e-21,
                                 1.694065894509399669649398521836e-21, 4.235164736272389463688418879636e-22,
                                 1.058791184067974064762782460584e-22, 2.646977960169798160618902050189e-23,
                                 6.617444900424343177893912768629e-24, 1.654361225106068880734221123349e-24,
                                 4.135903062765153408791935838694e-25, 1.033975765691286264082026643327e-25,
                                 2.584939414228213340076225223666e-26, 6.462348535570530772269628236053e-27,
                                 1.615587133892632406631747637268e-27, 4.038967834731580698317525293132e-28,
                                 1.009741958682895139216954234507e-28, 2.524354896707237808750799932127e-29,
                                 6.310887241768094478219682436680e-30, 1.577721810442023614704107565240e-30,
                                 3.944304526105059031370476640000e-31, 9.860761315262647572437533499000e-32,
                                 2.465190328815661892443976898000e-32, 6.162975822039154730370601500000e-33,
                                 1.540743955509788682510501190000e-33, 3.851859888774471706184973900000e-34,
                                 9.629649721936179265360991000000e-35, 2.407412430484044816328953000000e-35,
                                 6.018531076210112040809600000000e-36, 1.504632769052528010200750000000e-36,
                                 3.761581922631320025497600000000e-37, 9.403954806578300063715000000000e-38,
                                 2.350988701644575015901000000000e-38, 5.877471754111437539470000000000e-39,
                                 1.469367938527859384580000000000e-39, 3.673419846319648458500000000000e-40,
                                 9.183549615799121117000000000000e-41, 2.295887403949780249000000000000e-41,
                                 5.739718509874450320000000000000e-42, 1.434929627468612270000000000000e-42
        } ;

        double sum=0. ;

        /* Abramowitz-Stegun 27.1.1 */
        const double x2pi=x*M_1_2PI ;
        int k;
        for(k=1;k< sizeof(koeff)/sizeof(double)-1 ;k++)
        {
            const double sumold=sum ;
            /* do not precompute x2pi^2 to avoid loss of precision */
            sum += (2.+koeff[k])*pow(x2pi,2.*k)/(2*k+n) ;
            k++ ;
            sum -= (2.+koeff[k])*pow(x2pi,2.*k)/(2*k+n) ;
            if(sum == sumold)
                break ;
        }
        sum += 1./n-x/(2*(1+n)) ;
        return sum*pow(x,(double)n) ;
    }
}

#undef M_1_2PI