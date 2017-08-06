#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_linalg.h>
#define FALSE 0
#define TRUE  1
#define DATAFILE "data.txt"
#define DATALIENFILE "datalien.txt"
#define POPULATION_SIZE 100
#define GENERATION_COUNT 100000

/// Déclarations des fonctions
void init_ga();
int nbLignes(FILE*, char []);
int batch_memory_allocation();
void batch_memory_free();
int allocate_memory_double(double***, int, int);
void free_memory_double(double***, int);
int allocate_memory_int(int***, int, int);
void free_memory_int(int***, int);
int matrix_fill();
void A_fill();
void reg_mat_compute(double**, double*, int, int);
void K_assembly(double**, double**, int, int);
void K_reduce(double** , double**);
void dep_vector(double**, gsl_matrix*, gsl_vector*, gsl_vector*);
double deformation_energy(double*);
double def_energy_compute(double*, gsl_vector*);
void ga_doOneRun();
void select_best(int*, double*);
void generate_population_method_1();
void generate_population_method_2();
void choose_from_sample(double**, int);
void generator(double*);
void normalize(double*);
double r2();
void sort(double*, int);
double constraint_check(double*);
void set_TRIVIAL_CHROMOSOM_1(double*);
void set_TRIVIAL_CHROMOSOM_2(double*);
void MUTATE(double*, double*);
void generate_next_generation();
void select_parents(double*, int*, int*);
void sort_inv_score(double*, int*);
void CROSSOVER(double*, double*, double, double);
void bread_crumbs(int);



/// Déclarations des variables globales
int N, Nc;
const int datacols = 6, dataliencols = 2, Acols = 3;
double** data;
int** datalien;
int** conMat;
double** A;
int* ddl;
double* F;
double* F_red;
int nddl;
int lCount;
const double E = 200000;
FILE* f = NULL;
double** POPULATION;
int CHROMOSOM_SIZE;
const int MUTATION_PARAMETER = 5;
int verbose = FALSE;


int main()
{
    printf("************************** GA **************************\n");

    printf("\nAppuyer sur ENTRER pour commencer l'initialisation\n");
    getchar();

    /// ------------------------ Initialisation ------------------------
    init_ga();
    /// ----------------------------------------------------------------


    printf("\nAppuyer sur ENTRER pour demarrer l'algorithme genetique..\n");
    getchar();


    /// ------------------------- genitic algo -------------------------
    ga_doOneRun(); /// GA
    /// ----------------------------------------------------------------

    /// ------------------------- free memory --------------------------
    batch_memory_free();
    /// ----------------------------------------------------------------


    return 0;
}

void init_ga() {

    ///calcul de nb de lignes
    printf("\n1. Calcul nombre de lignes.. ");
    N = nbLignes(f, DATAFILE);
    Nc = nbLignes(f, DATALIENFILE);
    if (N == -1 || Nc == -1) {
        printf("\n\t[ERREUR][ECHEC_OUV_FICH]");
        exit(1);
    }
    printf("[DONE]");

    ///Allocation dynamique des matrices
    printf("\n2. allocation de memoire.. ");
    int allocation_res = batch_memory_allocation();
    if (allocation_res == FALSE) {
        printf("\n\t[ERREUR][MEM_INSUF]");
        exit(1);
    }
    printf("[DONE]");

    ///Remplissage des matrices à partir des fichiers
    printf("\n3. Remplissage des matrices a partir des fichiers.. ");
    int fill_res = matrix_fill();
    if (fill_res == FALSE) {
        printf("\n\t[ERREUR][ECHEC_OUV_FICH]");
        exit(1);
    }
    printf("[DONE]");

    ///Calcul de la matrice de connectivité
    printf("\n5. Calcul de la matrice de connectivite.. ");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            conMat[i][j] = FALSE;
        }
    }
    int temp0, temp1;
    for (int i = 0; i < Nc; i++) {
            temp0 = datalien[i][0];
            temp1 = datalien[i][1];
            conMat[temp0][temp1] = TRUE;
    }
    printf("[DONE]");

    ///Calcul du nombre de liaisons
    printf("\n6. Calcul du nombre de liaisons.. ");
    lCount = 0;
    for (int i = 0; i < N-1; i++) {
        for (int j = i+1; j < N; j++) {
            if (conMat[i][j] == TRUE) lCount++;
        }
    }
    printf("[DONE]");

    ///Création de la matrice A
    printf("\n7. Creation de la matrice A.. ");
    allocation_res = allocate_memory_double(&A, lCount, Acols);
    if (allocation_res == FALSE) {
        printf("\n\t[ERREUR][MEM_INSUF]");
        exit(1);
    }
    A_fill();
    printf("[DONE]");

    ///Calcul des ddl, nddl, F et F_red
    printf("\n8. Calcul des ddl, nddl, F et F_red.. ");
	int index_i, index_j;
	ddl = (int*)malloc(2*N * sizeof(int));
	if (ddl == NULL) {
        printf("\n\t[ERREUR][MEM_INSUF]");
        exit(1);
	}
	for (int i = 0; i < 2*N; i++) {
        index_i = (int)(i / 2);
        if (i % 2 == 0)
            index_j = 4;
        else
            index_j = 5;
        ddl[i] = data[index_i][index_j];
    }
    nddl = 0;
	for (int i = 0; i < 2*N; i++) {
		if (ddl[i] == 1)
			nddl++;
	}
    F = (double*)malloc(2*N * sizeof(double));
	for (int i = 0; i < 2*N; i++) {
        index_i = (int)(i / 2);
        if (i % 2 == 0)
            index_j = 2;
        else
            index_j = 3;
        F[i] = data[index_i][index_j];
    }
	F_red = (double*)malloc(nddl * sizeof(double));
	index_i = 0;
	for (int i = 0; i < 2*N; i++) {
        if (ddl[i] == 1) {
            F_red[index_i] = F[i];
            index_i++;
        }
	}
	printf("[DONE]");

    verbose = FALSE;
    bread_crumbs(verbose);


}

void bread_crumbs(int verbose) {
    if (verbose == FALSE) {
        printf("\nNb de lignes:\n\tdata : %d\tdatalien lignes : %d\n", N, Nc);
        printf("\n\tNombre de liasons : %d\n", lCount);
        printf("\n\nFIN DE L'INITIALISATION\n");
    } else {
        printf("\n--------------------- VERBOSE -----------------------\n");
        printf("\nNb de lignes:\n\tdata : %d\tdatalien lignes : %d\n", N, Nc);
        printf("\nAffichage des matrices :");
        printf("\n\tdata\n\t\t");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < datacols; j++) {
                printf("%.2f\t", data[i][j]);
            }
            printf("\n\t\t");
        }
        printf("\n\tdatalien\n\t\t");
        for (int i = 0; i < Nc; i++) {
            for (int j = 0; j < dataliencols; j++) {
                printf("%d\t", datalien[i][j]);
            }
            printf("\n\t\t");
        }
        printf("\nAffichage de la mat de connectivite :\n\t");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d\t", conMat[i][j]);
            }
            printf("\n\t");
        }
        printf("\n\tNombre de liasons : %d\n", lCount);
        printf("\nAffichage de A:\n\t");
        for (int i = 0; i < lCount; i++) {
            for (int j = 0; j < Acols; j++) {
                printf("%.2f\t", A[i][j]);
            }
            printf("\n\t");
        }

        printf("\n-----------ddl-------------");
        for (int i = 0; i < 2*N; i++) {
            printf("\n\t%d", ddl[i]);
        }
        printf("\nnddl = %d", nddl);
        printf("\n-----------Fex-------------");
        for (int i = 0; i < 2*N; i++) {
            printf("\n\t%lf", F[i]);
        }
        printf("\n----------F_red------------");
        for (int i = 0; i < nddl; i++) {
            printf("\n\t%lf", F_red[i]);
        }
        printf("\n\nFIN DE L'INITIALISATION\n");
    }
}

int nbLignes(FILE* f, char fileName []) {
    f = fopen(fileName, "r");
    if (f != NULL) {
        char ch;
        int N = 1; //nb de lignes
        do {
            ch = fgetc(f);
            if(ch == '\n') N++;
        }
        while (ch != EOF);
        fclose(f);
        return(N);
    }
    return(-1);
}

int batch_memory_allocation() {
    int dt = allocate_memory_double(&data, N, datacols);
    int dtl = allocate_memory_int(&datalien, Nc, dataliencols);
    int cnm = allocate_memory_int(&conMat, N, N);
    if (dt == FALSE || dtl == FALSE || cnm == FALSE)
        return FALSE;
    return TRUE;
}

void batch_memory_free() {
    free_memory_double(&data, N);
    free_memory_int(&datalien, Nc);
    free_memory_int(&conMat, N);
    free_memory_double(&A, lCount);
    free(ddl);
    free(F);
    free(F_red);
}

int allocate_memory_double(double*** mat, int nrows, int ncols) {
    *mat = (double**)malloc(nrows * sizeof(double*));
    if (*mat == NULL) return FALSE;
    else {
        for (int i = 0; i < nrows; i++) {
            (*mat)[i] = (double*)malloc(ncols * sizeof(double));
            if ((*mat)[i] == NULL) return FALSE;
        }
    }
    return TRUE;
}

void free_memory_double(double*** mat, int nrows) {
    for (int i = 0; i < nrows; i++) {
        free((*mat)[i]);
    }
    free(*mat);
}

int allocate_memory_int(int*** mat, int nrows, int ncols) {
    *mat = (int**)malloc(nrows * sizeof(int*));
    if (*mat == NULL) return FALSE;
    else {
        for (int i = 0; i < nrows; i++) {
            (*mat)[i] = (int*)malloc(ncols * sizeof(int));
            if ((*mat)[i] == NULL) return FALSE;
        }
    }
    return TRUE;
}

void free_memory_int(int*** mat, int nrows) {
    for (int i = 0; i < nrows; i++) {
        free((*mat)[i]);
    }
    free(*mat);
}

int matrix_fill() {
    f = fopen(DATAFILE, "r");
    if (f != NULL) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < datacols; j++) {
                double x;
                fscanf(f, "%lf", &x);
                data[i][j] = x;
            }
        }
    } else return FALSE;
    fclose(f);
    f = fopen(DATALIENFILE, "r");
    if (f != NULL) {
        for (int i = 0; i < Nc; i++) {
            for (int j = 0; j < dataliencols; j++) {
                double x;
                fscanf(f, "%lf", &x);
                datalien[i][j] = x;
            }
        }
    } else return FALSE;
    fclose(f);
    return TRUE;
}

void A_fill() {
    int counter = 0;
    double xi;
    double xj;
    double yi;
    double yj;
    for (int i = 0; i < N-1; i++) {
        for (int j = i+1; j < N; j++) {
            if (conMat[i][j] == TRUE) {
                A[counter][0] = i;
                A[counter][1] = j;
                xi = data[i][0];
                xj = data[j][0];
                yi = data[i][1];
                yj = data[j][1];
                A[counter][2] = sqrt(pow(xi - xj, 2) + pow(yi - yj, 2));
                counter++;
            }
        }
    }
}

void reg_mat_compute(double** K, double* S, int N, int lCount) {
    double** Ke_block;
    int allocation_res = allocate_memory_double(&Ke_block, 2, 2);
    if (allocation_res == FALSE) {
        printf("\n\t[ERREUR][MEM_INSUF]");
        exit(1);
    }
    for (int i = 0; i < 2*N; i++) {
        for (int j = 0; j < 2*N; j++) {
            K[i][j] = 0;
        }
    }
    int p, i, j;
    double L, nx, ny;
    for (p = 0; p < lCount; p++) {
        i = (int)(A[p][0]);
        j = (int)(A[p][1]);
        L = A[p][2];
        nx = (data[j][0] - data[i][0]) / L;
        ny = (data[j][1] - data[i][1]) / L;
//        printf("\n\nINSIDE REG_MAT_COMPUTE P = %d>>>", p);
//        printf("\n\ti:%d\tj:%d\tL:%lf\tnx:%lf\tny:%lf", i, j, L, nx, ny);
        Ke_block[0][0] = E * S[p] * nx * nx / L;
        Ke_block[0][1] = E * S[p] * nx * ny / L;
        Ke_block[1][0] = E * S[p] * nx * ny / L;
        Ke_block[1][1] = E * S[p] * ny * ny / L;
//        printf("\n\t%lf\t%lf\n\t%lf\t%lf", Ke_block[0][0], Ke_block[0][1], Ke_block[1][0], Ke_block[1][1]);
        K_assembly(K, Ke_block, 2*i, 2*i);
        K_assembly(K, Ke_block, 2*i, 2*j);
        K_assembly(K, Ke_block, 2*j, 2*i);
        K_assembly(K, Ke_block, 2*j, 2*j);
    }
    free_memory_double(&Ke_block, 2);
}

void K_assembly(double** K, double** Ke_block, int row, int col) {
    int sign;
    if (row == col)
        sign = 1;
    else
        sign = -1;
    for (int i = row; i <= row + 1; i++) {
        for (int j = col; j <= col + 1; j++) {
            K[i][j] += sign * Ke_block[i%2][j%2];
        }
    }
}

double deformation_energy(double* S) {
    double** K;
    int allocation_res = allocate_memory_double(&K, 2*N, 2*N);
    if (allocation_res == FALSE) {
        printf("\n\t[ERREUR][MEM_INSUF]");
        exit(1);
    }
    reg_mat_compute(K, S, N, lCount);

//    printf("\n\n\n-----------------------MAT DE REG-----------------------\n");
//    for (int i = 0; i < 2*N; i++) {
//        for (int j = 0; j < 2*N; j++) {
//            printf("%lf\t", K[i][j]);
//        }
//        printf("\n");
//    }

    double** K_red;
    allocation_res = allocate_memory_double(&K_red, nddl, nddl);
    if (allocation_res == FALSE) {
        printf("\n\t[ERREUR][MEM_INSUF]");
        exit(1);
    }
    K_reduce(K, K_red);

//    printf("\n\n\n---------------------MAT DE REG RED---------------------\n");
//    for (int i = 0; i < nddl; i++) {
//        for (int j = 0; j < nddl; j++) {
//            printf("%lf\t", K_red[i][j]);
//        }
//        printf("\n");
//    }

//    printf("\n\n\n---------------------VECTEUR DE DEP---------------------\n");
    ///resolution de mx=b avec m = K_red et b = F_red
    gsl_matrix* m = gsl_matrix_alloc(nddl, nddl);
    gsl_vector* b = gsl_vector_alloc(nddl);
    gsl_vector* x = gsl_vector_alloc(nddl);
    dep_vector(K_red, m, b, x);
//    gsl_vector_fprintf (stdout, x, "%g");

//    printf("\n\n\n---------------------ENERGIE DE DEF---------------------\n");
    double def_energy;
    def_energy = def_energy_compute(F_red, x);

    gsl_matrix_free(m);
    gsl_vector_free(b);
    gsl_vector_free(x);
    free_memory_double(&K, 2*N);
    free_memory_double(&K_red, nddl);

    return def_energy;
}

void K_reduce(double** K, double** K_red) {
    int i, j, index_i, index_j;

    index_i = 0;
    for (i = 0; i < 2*N; i++) {
        if (ddl[i] == 1) {
            index_j = 0;
            for (j = 0; j < 2*N; j++) {
                if (ddl[j] == 1) {
                    K_red[index_i][index_j] = K[i][j];
                    index_j++;
                }
            }
            index_i++;
        }
    }
}

void dep_vector(double** K_red, gsl_matrix* m, gsl_vector* b, gsl_vector* x) {
    for (int i = 0; i < nddl; i++) {
        for (int j = 0; j < nddl; j++) {
            gsl_matrix_set(m, i, j, K_red[i][j]);
        }
    }
    for (int i = 0; i < nddl; i++)
        gsl_vector_set(b, i, F_red[i]);

    int s;
    gsl_permutation * p = gsl_permutation_alloc (nddl);
    gsl_linalg_LU_decomp (m, p, &s);
    gsl_linalg_LU_solve (m, p, b, x);

    gsl_permutation_free (p);
}

double def_energy_compute(double* F_red, gsl_vector* x) {
    double def_energy = 0;
    for (int i = 0; i < nddl; i++) {
        def_energy += F_red[i] * gsl_vector_get(x, i);
    }
    def_energy *= 0.5;
    return (def_energy);
}

void ga_doOneRun() {
    int allocation_res;
    srand(time(NULL));
    CHROMOSOM_SIZE = lCount;
    allocation_res = allocate_memory_double(&POPULATION, POPULATION_SIZE, CHROMOSOM_SIZE);
    if (allocation_res == FALSE) {
        printf("\n\t[ERREUR][MEM_INSUF]");
        exit(1);
    }

    ///generate population
    generate_population_method_2();


    ///loop:
    for (int GENERATION = 2; GENERATION <= GENERATION_COUNT; GENERATION++) {
        if (GENERATION % 1000 == 0)
            printf("\nGENERATION %d", GENERATION);
        generate_next_generation();
    }


    ///best individual
    int BEST_INDIVIDUAL;
    double* score;
    score = (double*)malloc(POPULATION_SIZE * sizeof(double));
    if (score == NULL) {
        printf("\n\t[ERREUR][MEM_INSUF]");
        exit(1);
    }
    select_best(&BEST_INDIVIDUAL, score);
    printf("\n\nBEST INDIVIDUAL: %d", BEST_INDIVIDUAL);
    for (int i = 0; i < CHROMOSOM_SIZE; i++) {
        printf("\n%lf", POPULATION[BEST_INDIVIDUAL][i]);
    }
    printf("\n\nCONSTRAINT CHECK:\n");
    printf("SUM (Li*Si) = %lf", constraint_check(POPULATION[BEST_INDIVIDUAL]));
    printf("\n\nBEST DEFORMATION ENERGY SCORE: %lf\n", score[BEST_INDIVIDUAL]);

    free_memory_double(&POPULATION, POPULATION_SIZE);
    free(score);

}

void select_best(int* min_index, double* score) {
    double min_score;

    for (int CHROMOSOM = 0; CHROMOSOM < POPULATION_SIZE; CHROMOSOM++) {
        score[CHROMOSOM] = deformation_energy(POPULATION[CHROMOSOM]);
    }

    *min_index = 0;
    min_score = score[0];
    for (int CHROMOSOM = 0; CHROMOSOM < POPULATION_SIZE; CHROMOSOM++) {
        if (score[CHROMOSOM] < min_score) {
            *min_index = CHROMOSOM;
            min_score = score[CHROMOSOM];
        }
    }
}

void generate_population_method_1() {
    int SAMPLE_SIZE = POPULATION_SIZE * 5;
    double** SAMPLE_POPULATION;

    double* TRIVIAL_CHROMOSOM_1;
    double* TRIVIAL_CHROMOSOM_2;
    double* MODIFIED_CHROMOSOM_1;
    double* MODIFIED_CHROMOSOM_2;
    TRIVIAL_CHROMOSOM_1 = (double*)malloc(CHROMOSOM_SIZE * sizeof(double));
    MODIFIED_CHROMOSOM_1 = (double*)malloc(CHROMOSOM_SIZE * sizeof(double));
    TRIVIAL_CHROMOSOM_2 = (double*)malloc(CHROMOSOM_SIZE * sizeof(double));
    MODIFIED_CHROMOSOM_2 = (double*)malloc(CHROMOSOM_SIZE * sizeof(double));
    set_TRIVIAL_CHROMOSOM_1(TRIVIAL_CHROMOSOM_1);
    set_TRIVIAL_CHROMOSOM_2(TRIVIAL_CHROMOSOM_2);

    printf("\n\nTRIVIAL_CHROMOSOM_1:\n");
    for (int i = 0; i < CHROMOSOM_SIZE; i++) {
        printf("\t%lf", TRIVIAL_CHROMOSOM_1[i]);
    }
    printf("\n\nCONSTRAINT CHECK:\n");
    printf("SUM LiSi = %lf\n", constraint_check(TRIVIAL_CHROMOSOM_1));

    printf("\n\nTRIVIAL_CHROMOSOM_2:\n");
    for (int i = 0; i < CHROMOSOM_SIZE; i++) {
        printf("\t%lf", TRIVIAL_CHROMOSOM_2[i]);
    }
    printf("\n\nCONSTRAINT CHECK:\n");
    printf("SUM LiSi = %lf\n", constraint_check(TRIVIAL_CHROMOSOM_2));

    MUTATE(TRIVIAL_CHROMOSOM_1, MODIFIED_CHROMOSOM_1);
    printf("\n\nMODIFIED_CHROMOSOM_1:\n");
    for (int i = 0; i < CHROMOSOM_SIZE; i++) {
        printf("\t%lf", MODIFIED_CHROMOSOM_1[i]);
    }
    printf("\n\nCONSTRAINT CHECK:\n");
    printf("SUM LiSi = %lf\n", constraint_check(MODIFIED_CHROMOSOM_1));

    MUTATE(TRIVIAL_CHROMOSOM_2, MODIFIED_CHROMOSOM_2);
    printf("\n\nMODIFIED_CHROMOSOM_2:\n");
    for (int i = 0; i < CHROMOSOM_SIZE; i++) {
        printf("\t%lf", MODIFIED_CHROMOSOM_2[i]);
    }
    printf("\n\nCONSTRAINT CHECK:\n");
    printf("SUM LiSi = %lf\n", constraint_check(MODIFIED_CHROMOSOM_2));


    allocate_memory_double(&SAMPLE_POPULATION, SAMPLE_SIZE, CHROMOSOM_SIZE);
    int RANDOM_TRIVIAL_CHROMOSOM;
    for (int CHROMOSOM = 0; CHROMOSOM < SAMPLE_SIZE; CHROMOSOM++) {
        RANDOM_TRIVIAL_CHROMOSOM = rand() % 2;
        if (RANDOM_TRIVIAL_CHROMOSOM == 0)
            MUTATE(TRIVIAL_CHROMOSOM_1, MODIFIED_CHROMOSOM_1);
        else
            MUTATE(TRIVIAL_CHROMOSOM_2, MODIFIED_CHROMOSOM_2);
        for (int GENE = 0; GENE < CHROMOSOM_SIZE; GENE++) {
            if (RANDOM_TRIVIAL_CHROMOSOM == 0)
                SAMPLE_POPULATION[CHROMOSOM][GENE] = MODIFIED_CHROMOSOM_1[GENE];
            else
                SAMPLE_POPULATION[CHROMOSOM][GENE] = MODIFIED_CHROMOSOM_2[GENE];
            TRIVIAL_CHROMOSOM_1[GENE] = MODIFIED_CHROMOSOM_1[GENE];
            TRIVIAL_CHROMOSOM_2[GENE] = MODIFIED_CHROMOSOM_2[GENE];
        }
    }

    choose_from_sample(SAMPLE_POPULATION, SAMPLE_SIZE);

    free(TRIVIAL_CHROMOSOM_1);
    free(TRIVIAL_CHROMOSOM_2);
    free(MODIFIED_CHROMOSOM_1);
    free(MODIFIED_CHROMOSOM_2);
    free_memory_double(&SAMPLE_POPULATION, SAMPLE_SIZE);

    printf("\n\nSELECTED POPULATION :\n");
    for (int CHROMOSOM = 0; CHROMOSOM < POPULATION_SIZE; CHROMOSOM++) {
        printf("\nCHROMOSOM %d:\n", CHROMOSOM);
        for (int GENE = 0; GENE < CHROMOSOM_SIZE; GENE++) {
            printf("\t%lf", POPULATION[CHROMOSOM][GENE]);
        }
        printf("\nCONSTRAINT CHECK:\n");
        printf("SUM LiSi = %lf\n", constraint_check(POPULATION[CHROMOSOM]));
    }
}

void generate_population_method_2() {
    int SAMPLE_SIZE = POPULATION_SIZE * 5;
    double** SAMPLE_POPULATION;
    allocate_memory_double(&SAMPLE_POPULATION, SAMPLE_SIZE, CHROMOSOM_SIZE);
    double* NEXT_CHROMOSOM;
    NEXT_CHROMOSOM = (double*)malloc(CHROMOSOM_SIZE * sizeof(double));

    for (int CHROMOSOM = 0; CHROMOSOM < SAMPLE_SIZE; CHROMOSOM++) {
        generator(NEXT_CHROMOSOM);
        for (int GENE = 0; GENE < CHROMOSOM_SIZE; GENE++) {
            SAMPLE_POPULATION[CHROMOSOM][GENE] = NEXT_CHROMOSOM[GENE];
        }
    }

    choose_from_sample(SAMPLE_POPULATION, SAMPLE_SIZE);

    free(NEXT_CHROMOSOM);
    free_memory_double(&SAMPLE_POPULATION, SAMPLE_SIZE);

    printf("\n\nSELECTED POPULATION :\n");
    for (int CHROMOSOM = 0; CHROMOSOM < POPULATION_SIZE; CHROMOSOM++) {
        printf("CHROMOSOM %d:", CHROMOSOM);
        for (int GENE = 0; GENE < CHROMOSOM_SIZE; GENE++) {
            printf("\t%lf", POPULATION[CHROMOSOM][GENE]);
        }
        printf("\t| CONSTRAINT CHECK: ");
        printf("SUM = %lf\n", constraint_check(POPULATION[CHROMOSOM]));
    }

}

void choose_from_sample(double** SAMPLE_POPULATION, int SAMPLE_SIZE) {
    int* SELECTION_LOG;
    SELECTION_LOG = (int*)malloc(SAMPLE_SIZE * sizeof(int));
    for (int i = 0; i < SAMPLE_SIZE; i++)
        SELECTION_LOG[i] = FALSE;

    int SELECTED_CHROMOSOM;
    for (int CHROMOSOM = 0; CHROMOSOM < POPULATION_SIZE; CHROMOSOM++) {
        do
            SELECTED_CHROMOSOM = rand() % SAMPLE_SIZE;
        while (SELECTION_LOG[SELECTED_CHROMOSOM] == TRUE);
        for (int GENE = 0; GENE < CHROMOSOM_SIZE; GENE++) {
            POPULATION[CHROMOSOM][GENE] = SAMPLE_POPULATION[SELECTED_CHROMOSOM][GENE];
        }
        SELECTION_LOG[SELECTED_CHROMOSOM] = TRUE;
    }

    free(SELECTION_LOG);
}

void generator(double* GENERATED_CHROMOSOM) {
    int RAN_VEC_SIZE = CHROMOSOM_SIZE - 1;
	double* RANDOM_VECTOR;
    RANDOM_VECTOR = (double*)malloc(RAN_VEC_SIZE*sizeof(double));

	for (int i = 0; i < RAN_VEC_SIZE; i++) {
        do
            RANDOM_VECTOR[i] = r2();
        while (RANDOM_VECTOR[i] == 0);
	}

	sort(RANDOM_VECTOR, RAN_VEC_SIZE);

	for (int i = 1; i < RAN_VEC_SIZE; i++) {
		GENERATED_CHROMOSOM[i] = RANDOM_VECTOR[i] - RANDOM_VECTOR[i-1];
	}

	GENERATED_CHROMOSOM[0] = RANDOM_VECTOR[0];
	GENERATED_CHROMOSOM[RAN_VEC_SIZE] = 1 - RANDOM_VECTOR[RAN_VEC_SIZE - 1];

    normalize(GENERATED_CHROMOSOM);
    free(RANDOM_VECTOR);
}

void normalize(double* GENERATED_CHROMOSOM) {
    for (int GENE = 0; GENE < CHROMOSOM_SIZE; GENE++) {
        GENERATED_CHROMOSOM[GENE] = GENERATED_CHROMOSOM[GENE] / A[GENE][2];
    }
}

double r2()
{
    return (double)rand() / (double)RAND_MAX;
}

void sort(double* RANDOM_VECTOR, int RAN_VEC_SIZE) {
	double temp;
	for (int i = 0; i < RAN_VEC_SIZE; i++) {
		for (int j = i + 1; j < RAN_VEC_SIZE; j++) {
			if (RANDOM_VECTOR[i] > RANDOM_VECTOR[j]) {
				temp = RANDOM_VECTOR[i];
				RANDOM_VECTOR[i] = RANDOM_VECTOR[j];
				RANDOM_VECTOR[j] = temp;
			}
		}
	}
}

double constraint_check(double* S) {
    double sum = 0;
    for (int i = 0; i < CHROMOSOM_SIZE; i++) {
        sum += S[i] * A[i][2];
    }
    return sum;
}

void set_TRIVIAL_CHROMOSOM_1(double* TRIVIAL_CHROMOSOM_1) {
    for (int i = 0; i < CHROMOSOM_SIZE; i++) {
        TRIVIAL_CHROMOSOM_1[i] = 1 / (CHROMOSOM_SIZE * A[i][2]);
    }
}

void set_TRIVIAL_CHROMOSOM_2(double* TRIVIAL_CHROMOSOM_2) {
    double sum = 0;
    for (int i = 0; i < CHROMOSOM_SIZE; i++) {
        sum += A[i][2];
    }
    for (int i = 0; i < CHROMOSOM_SIZE; i++) {
        TRIVIAL_CHROMOSOM_2[i] = 1 / sum;
    }
}

void MUTATE(double* ORIGINAL_CHROMOSOM, double* MUTATED_CHROMOSOM) {
    ///TWO POINT MUTATION
    int FIRST_GENE, SECOND_GENE, ALPHA;
    ALPHA = (rand() % MUTATION_PARAMETER) + 2; /// = 2 || 3 || 4 || 5 || 6
    FIRST_GENE = rand() % CHROMOSOM_SIZE;
    do
        SECOND_GENE = rand() % CHROMOSOM_SIZE;
    while (FIRST_GENE == SECOND_GENE);
    for (int i = 0; i < CHROMOSOM_SIZE; i++)
        MUTATED_CHROMOSOM[i] = ORIGINAL_CHROMOSOM[i];
    MUTATED_CHROMOSOM[FIRST_GENE] =
        (ALPHA - 1) * ORIGINAL_CHROMOSOM[FIRST_GENE] / ALPHA;
    MUTATED_CHROMOSOM[SECOND_GENE] =
        ORIGINAL_CHROMOSOM[FIRST_GENE] * A[FIRST_GENE][2] / (ALPHA * A[SECOND_GENE][2]) + ORIGINAL_CHROMOSOM[SECOND_GENE];
}

void generate_next_generation() {
    double* score;
    score = (double*)malloc(POPULATION_SIZE * sizeof(double));

    for (int CHROMOSOM = 0; CHROMOSOM < POPULATION_SIZE; CHROMOSOM++) {
        score[CHROMOSOM] = deformation_energy(POPULATION[CHROMOSOM]);
    }

    int SELECTED_PARENT_1, SELECTED_PARENT_2;
    select_parents(score, &SELECTED_PARENT_1, &SELECTED_PARENT_2);

    if (verbose == TRUE) {
        printf("\n\nPOPULATION DEFORMATION ENERGY:");
        for (int CHROMOSOM = 0; CHROMOSOM < POPULATION_SIZE; CHROMOSOM++) {
            printf("\nDeformation energy of CHROMOSOM %d : %lf", CHROMOSOM, score[CHROMOSOM]);
        }
        printf("\n");
        printf("\n\nSELECTED INDIVIDUALS:");
        printf("\nCHROMOSOM %d & CHROMOSOM %d\n", SELECTED_PARENT_1, SELECTED_PARENT_2);
    }

    CROSSOVER(POPULATION[SELECTED_PARENT_1], POPULATION[SELECTED_PARENT_2],
              score[SELECTED_PARENT_1], score[SELECTED_PARENT_2]);

    for (int CHROMOSOM = 0; CHROMOSOM < POPULATION_SIZE; CHROMOSOM++) {
        score[CHROMOSOM] = deformation_energy(POPULATION[CHROMOSOM]);
    }

    free(score);
}

void select_parents(double* score, int* PARENT_1, int* PARENT_2) {
    double* inv_score;
    int* index_log;
    double threshold, sum = 0;
    inv_score = (double*)malloc(POPULATION_SIZE * sizeof(double));
    index_log = (int*)malloc(POPULATION_SIZE * sizeof(int));

    for (int CHROMOSOM = 0; CHROMOSOM < POPULATION_SIZE; CHROMOSOM++) {
        inv_score[CHROMOSOM] = 1/score[CHROMOSOM];
        sum = sum + inv_score[CHROMOSOM];
    }

    for (int CHROMOSOM = 0; CHROMOSOM < POPULATION_SIZE; CHROMOSOM++) {
        inv_score[CHROMOSOM] = inv_score[CHROMOSOM]/sum;
    }

    sort_inv_score(inv_score, index_log);

    for (int CHROMOSOM = 1; CHROMOSOM < POPULATION_SIZE; CHROMOSOM++) {
        inv_score[CHROMOSOM] = inv_score[CHROMOSOM] + inv_score[CHROMOSOM - 1];
    }

    threshold = r2();
    for (int CHROMOSOM = 1; CHROMOSOM < POPULATION_SIZE; CHROMOSOM++) {
        if (inv_score[CHROMOSOM] >= threshold) {
            if (CHROMOSOM == POPULATION_SIZE - 1) {
                *PARENT_1 = index_log[CHROMOSOM];
                *PARENT_2 = index_log[CHROMOSOM - 1];
            } else {
                *PARENT_1 = index_log[CHROMOSOM + 1];
                *PARENT_2 = index_log[CHROMOSOM];
            }
            break;
        }
    }
    free(inv_score);
    free(index_log);
}

void sort_inv_score(double* inv_score, int* index_log) {
    for (int i = 0; i < POPULATION_SIZE; i++) {
        index_log[i] = i;
    }

    double temp_score;
    int temp_index;
	for (int i = 0; i < POPULATION_SIZE; i++) {
		for (int j = i + 1; j < POPULATION_SIZE; j++) {
			if (inv_score[i] > inv_score[j]) {
				temp_score = inv_score[i];
				inv_score[i] = inv_score[j];
				inv_score[j] = temp_score;

				temp_index = index_log[i];
				index_log[i] = index_log[j];
				index_log[j] = temp_index;
			}
		}
	}
}

void CROSSOVER(double* PARENT_1, double* PARENT_2, double score_1, double score_2) {
    double* CHILD_1;
    double* CHILD_2;
    double score_child_1, score_child_2;
    CHILD_1 = (double*)malloc(CHROMOSOM_SIZE * sizeof(double));
    CHILD_2 = (double*)malloc(CHROMOSOM_SIZE * sizeof(double));
    double ALPHA_1 = (double)rand()/(double)(RAND_MAX);
    double ALPHA_2 = (double)rand()/(double)(RAND_MAX);
    for (int GENE = 0; GENE < CHROMOSOM_SIZE; GENE++) {
        CHILD_1[GENE] = ALPHA_1 * PARENT_1[GENE] + (1 - ALPHA_1) * PARENT_2[GENE];
        CHILD_2[GENE] = (1 - ALPHA_2) * PARENT_1[GENE] + ALPHA_2 * PARENT_2[GENE];
    }
    score_child_1 = deformation_energy(CHILD_1);
    score_child_2 = deformation_energy(CHILD_2);

    if (verbose == TRUE) {
        printf("\n\nCROSSOVER:");
        printf("\nDeformation energy of CHILD_1: %lf", score_child_1);
        printf("\nDeformation energy of CHILD_2: %lf\n", score_child_2);

    }

    double* MUTATED_CHILD;
    if (score_child_1 > score_1 && score_child_1 > score_2) {
        MUTATED_CHILD = (double*)malloc(CHROMOSOM_SIZE * sizeof(double));
        MUTATE(CHILD_1, MUTATED_CHILD);
        for (int GENE = 0; GENE < CHROMOSOM_SIZE; GENE++) {
            CHILD_1[GENE] = MUTATED_CHILD[GENE];
        }
        free(MUTATED_CHILD);
    }
    if (score_child_2 > score_1 && score_child_2 > score_2) {
        MUTATED_CHILD = (double*)malloc(CHROMOSOM_SIZE * sizeof(double));
        MUTATE(CHILD_2, MUTATED_CHILD);
        for (int GENE = 0; GENE < CHROMOSOM_SIZE; GENE++) {
            CHILD_2[GENE] = MUTATED_CHILD[GENE];
        }
        free(MUTATED_CHILD);
    }

    score_child_1 = deformation_energy(CHILD_1);
    score_child_2 = deformation_energy(CHILD_2);

    for (int GENE = 0; GENE < CHROMOSOM_SIZE; GENE++) {
        PARENT_2[GENE] = CHILD_2[GENE];
    }
    for (int GENE = 0; GENE < CHROMOSOM_SIZE; GENE++) {
        PARENT_1[GENE] = CHILD_2[GENE];
    }

    free(CHILD_1);
    free(CHILD_2);

}
