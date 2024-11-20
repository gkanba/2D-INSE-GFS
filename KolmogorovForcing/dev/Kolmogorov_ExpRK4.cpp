/*
    @File       RestartGenerator_ExpRK4_Kolmogorov.cpp
    @Version    2024-11/20-v1.3
    @Author     Gakuto Kambayashi
    
    Author has all rights reserved.
*/

/*
    [Include directives] : C++ STL
*/ 
#include <cstdint>
#include <random>
#include <iostream>
#include <fstream>
#include <filesystem>

/*
    [Include directives] : OpenMP
*/
#include <omp.h>

/*
    [Include directives] : Mathematical Libraries (FFTW 3, Eigen 3)
*/
#include <fftw3.h>
#include <eigen3/Eigen/Dense>

/*
    [Include directives] : json Configulation Loader (nlohmann.json)
*/
#include <nlohmann/json.hpp>

/*
    [Constants] : Mathematical Constants
    PI  : Pi
    I   : Imaginary Unit
*/
constexpr double                PI = 3.141592653589793;
constexpr std::complex<double>  I(0.0, 1.0);

/*
    [Constants] I/O Formatter for Eigen Text output.
*/
static const Eigen::IOFormat    IOFMT(Eigen::FullPrecision, 0, ", ", "\n", "", "");

/*
    [Typedefs] : Typedef for Eigen RowMajor Matrices. ( !! We use RowMajor memory alignment for FFTW compatibility !! )
*/
typedef Eigen::Vector<double,               Eigen::Dynamic>                                  VectorRd;
typedef Eigen::Vector<std::complex<double>, Eigen::Dynamic>                                  VectorCd;
typedef Eigen::Matrix<double,               Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRd;
typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixCd;

/*
    [Struct - I/O] : Load Experiment Setup from .json Configuration.
*/
namespace Setup{
    struct JsonConfig{
        std::string     RootDir;                    // Root Directory
        int32_t         Threads;                    // Number of thread for OpenMP Parallelization

        double          Delta_t;                    // Temporal Discretization
        int32_t         N;                          // Domain Discretization ( N x N, FFT Size )

        double          Max_t;                      // Max              Time                          
        double          Out_t;                      // Output Cycle     Time                  
        double          Info_t;                     // Infomation Cycle Time              
        
        double          nu;                         // Kinematic Viscosity                   

        double          gamma;                      // Kolmogorov Forcing Amplitude                     
        int32_t         k;                          // Kolmogorov Forcing Wavenumber                    

        int32_t         Type;                       // Initial Condition type (0, 1, 2, 3)              
        int32_t         Seed_i;                     // Initial Condition seed (only used for type 3)    
    };

    void to_json(nlohmann::json& j, const JsonConfig& jc) {
        j = nlohmann::json{
            {"RootDir",         jc.RootDir},
            {"Threads",         jc.Threads},

            {"Delta_t",         jc.Delta_t},
            {"N",               jc.N},

            {"Max_t",           jc.Max_t},
            {"Out_t",           jc.Out_t},
            {"Info_t",          jc.Info_t},

            {"nu",              jc.nu},

            {"gamma",           jc.gamma},
            {"k",               jc.k},

            {"Type",            jc.Type},
            {"Seed_i",          jc.Seed_i},
        };
    }

    void from_json(const nlohmann::json& j, JsonConfig& jc){
        j.at("RootDir").get_to(jc.RootDir);
        j.at("Threads").get_to(jc.Threads);

        j.at("Delta_t").get_to(jc.Delta_t);
        j.at("N").get_to(jc.N);

        j.at("Max_t").get_to(jc.Max_t);
        j.at("Out_t").get_to(jc.Out_t);
        j.at("Info_t").get_to(jc.Info_t);

        j.at("nu").get_to(jc.nu);

        j.at("gamma").get_to(jc.gamma);
        j.at("k").get_to(jc.k);

        j.at("Type").get_to(jc.Type);
        j.at("Seed_i").get_to(jc.Seed_i);
    }
};

/*
    >> Preset values for experiment.
*/
struct Preset{
    
    // -- Domain
    double      Lx;             // x direction length                       ( CONSTANT = 2 * PI )
    double      Ly;             // y direction length                       ( CONSTANT = 2 * PI )
    int32_t     Nx;             // x direction grid count                   ( config.json initialized )
    int32_t     Ny;             // y direction grid count                   ( config.json initialized )
    int32_t     Nxp;            // y direction grid count padded            ( CONSTANT = Nx * 3 / 2 )
    int32_t     Nyp;            // y direction grid count padded            ( CONSTANT = Ny * 3 / 2 )

    // -- Params
    double      nu;             // kinematic viscousity                     ( config.json initialized )
    double      dt;             // temporal discretization                  ( config.json initialized )
    double      t_max;          // max          Time                        ( config.json initialized )
    double      t_out;          // output       Time                        ( config.json initialized )
    double      t_info;         // information  Time                        ( config.json initialized )

    // -- Flow, Forcing Initial Conditions
    double      gamma;          // kolmogorov Foricing Amplitude            ( config.json initialized )
    int32_t     k;              // kolmogorov Foricing Wavenumber           ( config.json initialized )
    int32_t     type;           // type for generating init cond            ( config.json initialized )
    int32_t     seed_i;         // seed for generating init cond            ( config.json initialized )

    // -- Fourier Wavenumbers
    VectorRd    kx;             // x direction wave numbers                 ( CONSTANT = Defined Later )
    VectorRd    ky;             // y direction wave numbers                 ( CONSTANT = Defined Later )
    MatrixRd    K;              // K(i, j)     = kx(i)^2 + ky(j)^2          ( CONSTANT = Defined Later )
    MatrixRd    K_inv;          // K_inv(i, j) = 1 / (kx(i)^2 + ky(j)^2)    ( CONSTANT = Defined Later )
                                // K_inv(0, 0) = 0.0
    // -- Mesh Grids
    VectorRd    X;              // x direction grid coordinates             ( CONSTANT = Defined Later )
    VectorRd    Y;              // y direction grid coordinates             ( CONSTANT = Defined Later )

    MatrixRd    X_m;            // x direction Mesh Grid                    ( CONSTANT = Defined Later )
    MatrixRd    Y_m;            // y direction Mesh Grid                    ( CONSTANT = Defined Later )

    int32_t     c_max() { return std::ceil(t_max / dt);  };     // time to Steps (Max)
    int32_t     c_out() { return std::ceil(t_out / dt);  };     // time to Steps (Out)
    int32_t     c_info(){ return std::ceil(t_info / dt); };     // time to Steps (Info)

};

/*
    [Struct - Data] : Stores data for Calculation ( Temporary matrices Allocation )
*/
struct Allocated{
    MatrixCd    lin_h;
    MatrixCd    adv_h;
    MatrixCd    u_h;
    MatrixCd    v_h;
    MatrixCd    psi_h;
    MatrixCd    domgdx_h;
    MatrixCd    domgdy_h;

    MatrixCd    adv_hp;
    MatrixCd    u_hp;
    MatrixCd    v_hp;
    MatrixCd    domgdx_hp;
    MatrixCd    domgdy_hp;

    MatrixRd    u_p;
    MatrixRd    v_p;
    MatrixRd    domgdx_p;
    MatrixRd    domgdy_p;

    MatrixRd    adv_p;
};

/*
    [Struct - Data] : Stores data for Physical Space ( Velocity Field, Stream Function, Vorticity Field )
*/
struct Velocities{
    MatrixRd    u;      // x direction Velocity Field               ( x-y Inverted ! Must be transposed before x-y plot ! )
    MatrixRd    v;      // y direction Velocity Field               ( x-y Inverted ! Must be transposed before x-y plot ! )    
    MatrixRd    psi;    // Stream Function      Field               ( x-y Inverted ! Must be transposed before x-y plot ! )
    MatrixRd    omg;    // Vorticity            Field               ( x-y Inverted ! Must be transposed before x-y plot ! )
};


/*
    [Functions - I/O] : Create Directory
    @path [std::string] - Relative file path to create from exec file location.
*/
bool CreateDir(std::string file_path){
    // -- Check if file_path is already exists
    if (std::filesystem::exists(file_path)) {
        std::cout << " -- Directory Already Exists  = " << file_path << std::endl;
        return false;
    }
    //-- Create directory
    bool isCreated = std::filesystem::create_directory(file_path);
    if(isCreated){
        std::cout << " -- Directory Created         = " << file_path << std::endl;
    }
    else{
        std::cout << " -- Failed to Create          = " << file_path << std::endl;
    }
    return isCreated;
}

/*
    [Functions - I/O] : Write Eigen Matrix data as a binary file.
    @dense      [const Eigen::PlainObjectBase<Derived>&] - Eigen Matrix object to save.
    @file_path  [std::string]                            - Relative file path to save data from exec file location.
*/
template<class Derived>
void WriteBinary(const Eigen::PlainObjectBase<Derived>& dense, std::string file_path){
    std::ofstream ofs(file_path.c_str(), std::ios_base::out | std::ios_base::binary);

    if(ofs.is_open()){
        // -- Get Matrix rows, cols
        int32_t rows = dense.rows();
        int32_t cols = dense.cols();
        // -- Write Matrix rows, cols       [First 8 Bytes : 2 Integer Data]
        ofs.write(reinterpret_cast<const char*>(&rows), sizeof(int32_t));
        ofs.write(reinterpret_cast<const char*>(&cols), sizeof(int32_t));
        // -- Write actual Matrix data
        ofs.write(reinterpret_cast<const char*>(dense.data()), sizeof(typename Derived::Scalar) * rows * cols);

        ofs.close();
        std::cout << " [FileSystem] Successfully saved -- " << file_path << std::endl;
    }
    else{
        std::cout << " [FileSystem] Failed to save -- " << file_path << std::endl;
    }

}

/*
    [Functions - I/O] : Load Eigen Matrix data from a binary file.
    @file_path  [std::string] - Relative file path to load data from exec file location.
*/
template<class Derived>
Derived LoadBinary(std::string file_path){
    Derived dense = typename Derived::Zero(1, 1);
    std::ifstream ifs(file_path.c_str(), std::ios_base::in | std::ios_base::binary);

    if(ifs.is_open()){
        // -- Read Matrix rows, cols    [First 8 Bytes : 2 Integer Data]
        int32_t rows, cols;
        ifs.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
        ifs.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
        // -- Resize Matrix for actual data
        dense.resize(rows, cols);
        // -- Read actual Matrix data
        ifs.read(reinterpret_cast<char*>(dense.data()), sizeof(typename Derived::Scalar) * rows * cols);

        ifs.close();
        std::cout << " [FileSystem] Successfully loaded -- " << file_path << std::endl;
    }
    else{
        std::cout << " [FileSystem] Failed to load -- " << file_path << std::endl;
    }

    return dense;
}

/*
    [Functions - I/O] : Write Eigen Matrix data as a text file.
    @dense      [const Eigen::PlainObjectBase<Derived>&] - Eigen Matrix object to save.
    @file_path  [std::string]                            - Relative file path to save data from exec file location.
*/
template<class Derived>
void WriteText(const Eigen::PlainObjectBase<Derived>& dense, std::string file_path, const Eigen::IOFormat format){
    std::ofstream ofs(file_path.c_str());
    if(ofs.is_open()){
        ofs << dense.format(format) << std::endl;
        ofs.close();
        std::cout << " [FileSystem] Successfully saved -- " << file_path << std::endl;
    }
    else{
        std::cout << " [FileSystem] Failed to save -- " << file_path << std::endl;
    }
}

/*
    [Struct - Math] : Object to carry out 2D Fast Fourier Transform for Real valued Matrix. ( !! Can only be used if physical values ensured to be real !! )
    @N0     [const int32_t]  - Transform size of first dimension.
    @N1     [const int32_t]  - Transform size of second dimension.
    @flag   [const unsigned] - FFTW Flags for FFTW plan.
*/
struct FFT2D_REAL {
    public:
        FFT2D_REAL(const int32_t N0, const int32_t N1, const unsigned flag) {
            _N0         = N0;
            _N1         = N1;
            _flag       = flag;
            _mat_in     = MatrixRd::Zero(_N0, _N1);
            _mat_out    = MatrixCd::Zero(_N0, _N1 / 2 + 1);
            _mat_ret    = MatrixCd::Zero(_N0, _N1);
            _ptr_in     = reinterpret_cast<double*>(_mat_in.data());
            _ptr_out    = reinterpret_cast<fftw_complex*>(_mat_out.data());
            _plan       = fftw_plan_dft_r2c_2d(_N0, _N1, _ptr_in, _ptr_out, _flag);
        };
        MatrixCd execute(MatrixRd& in) {
            // -- Set data pointers in, out
            _mat_in = in;
            _ptr_in = _mat_in.data();
            _ptr_out = reinterpret_cast<fftw_complex*>(_mat_out.data());

            // -- Execute Transform
            fftw_execute(_plan);

            // -- Copy Left     (Actual FFT Results)
            _mat_ret.leftCols(_N1 / 2 + 1) = _mat_out;
            // -- Top Right     (First Row)
            _mat_ret.row(0).segment(_N1 / 2 + 1, _N1 / 2 - 1) = _mat_ret.row(0).segment(1, _N1 / 2 - 1).conjugate().reverse();
            // -- Lower Right   (Remaining Rows)
            _mat_ret.bottomRows(_N0 - 1).rightCols(_N1 / 2 - 1) = _mat_ret.bottomRows(_N0 - 1).middleCols(1, _N1 / 2 - 1).conjugate().colwise().reverse().rowwise().reverse();

            // -- Normalization (Physically correct fourier coefficients)
            return (_mat_ret / (_N0 * _N1));
        }
        ~FFT2D_REAL(){
            fftw_destroy_plan(_plan);
        }
    public:
        int32_t         _N0;
        int32_t         _N1;
        unsigned        _flag;
        MatrixRd        _mat_in;
        MatrixCd        _mat_out;
        MatrixCd        _mat_ret;
        double*         _ptr_in;
        fftw_complex*   _ptr_out;
        fftw_plan       _plan;
};

/*
    [Struct - Math] : Object to carry out 2D Inverse Fast Fourier Transform for Complex valued Matrix. ( !! Can only be used if physical values ensured to be real !! )
    @N0     [const int32_t]  - Transform size of first dimension.
    @N1     [const int32_t]  - Transform size of second dimension.
    @flag   [const unsigned] - FFTW Flags for FFTW plan.
*/
struct IFFT2D_REAL {
    public:
        IFFT2D_REAL(const int32_t N0, const int32_t N1, const unsigned flag) {
            _N0         = N0;
            _N1         = N1;
            _flag       = flag;
            _mat_in     = MatrixCd::Zero(_N0, _N1 / 2 + 1);
            _mat_out    = MatrixRd::Zero(_N0, _N1);
            _ptr_in     = reinterpret_cast<fftw_complex*>(_mat_in.data());
            _ptr_out    = reinterpret_cast<double*>(_mat_out.data());
            _plan       = fftw_plan_dft_c2r_2d(_N0, _N1, _ptr_in, _ptr_out, _flag);
        };
        MatrixRd execute(const MatrixCd& in) {
             // -- Set data pointers in, out
            _mat_in  = in.leftCols(_N1 / 2 + 1);
            _ptr_in  = reinterpret_cast<fftw_complex*>(_mat_in.data());
            _ptr_out = _mat_out.data();

            // -- Execute Inverse Transform
            fftw_execute(_plan);

            // -- No Normalization here (Already Normalized when FFT carried out)
            // -- !! This means you cannot convert coefficients from external sources which is not normalized when their FFT is carried out !!
            return _mat_out;
        }
        ~IFFT2D_REAL(){
            fftw_destroy_plan(_plan);
        }
    public:
        int32_t         _N0;
        int32_t         _N1;
        unsigned        _flag;
        MatrixCd        _mat_in;
        MatrixRd        _mat_out;
        fftw_complex*   _ptr_in;
        double*         _ptr_out;
        fftw_plan       _plan;
};


/*
    [Functions - Calc] : Padding zero coefficients for 3/2-rule.
    @in     [const MatrixType&] - Matrix to be padded.
    @padded [const MatrixType&] - Matrix to store padded data.
    @ps     [std::string]       - Experiment Setups.
*/
template<typename MatrixType>
void pad(const MatrixType& in, MatrixType& padded, Preset& ps){
    
    // TODO : For loop and omp parallel for (might be faster)
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            padded.topLeftCorner(ps.Nx / 2 + 1, ps.Ny / 2 + 1)      = in.topLeftCorner(ps.Nx / 2 + 1, ps.Ny / 2 + 1);
        }
        #pragma omp section
        {
            padded.topRightCorner(ps.Nx / 2 + 1, ps.Ny / 2 - 1)     = in.topRightCorner(ps.Nx / 2 + 1, ps.Ny / 2 - 1);
        }
        #pragma omp section
        {
            padded.bottomLeftCorner(ps.Nx / 2 - 1, ps.Ny / 2 + 1)   = in.bottomLeftCorner(ps.Nx / 2 - 1, ps.Ny / 2 + 1);
        }
        #pragma omp section
        { 
            padded.bottomRightCorner(ps.Nx / 2 - 1, ps.Ny / 2 - 1)  = in.bottomRightCorner(ps.Nx / 2 - 1, ps.Ny / 2 - 1);
        }
    }

}

/*
    [Functions - Calc] : Chopping zero coefficients for 3/2-rule.
    @in     [const MatrixType&] - Matrix to be chopped.
    @padded [const MatrixType&] - Matrix to store chopped data.
    @ps     [std::string]       - Experiment Setups.
*/
template<typename MatrixType>
void chop(const MatrixType& in, MatrixType& chopped, Preset& ps){

    // TODO : For loop and omp parallel for (might be faster)
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            chopped.topLeftCorner(ps.Nx / 2 + 1, ps.Ny / 2 + 1)      = in.topLeftCorner(ps.Nx / 2 + 1, ps.Ny / 2 + 1);
        }
        #pragma omp section
        {
            chopped.topRightCorner(ps.Nx / 2 + 1, ps.Ny / 2 - 1)     = in.topRightCorner(ps.Nx / 2 + 1, ps.Ny / 2 - 1);
        }
        #pragma omp section
        {
            chopped.bottomLeftCorner(ps.Nx / 2 - 1, ps.Ny / 2 + 1)   = in.bottomLeftCorner(ps.Nx / 2 - 1, ps.Ny / 2 + 1);
        }
        #pragma omp section
        { 
            chopped.bottomRightCorner(ps.Nx / 2 - 1, ps.Ny / 2 - 1)  = in.bottomRightCorner(ps.Nx / 2 - 1, ps.Ny / 2 - 1);
        }
    }

}

/*
    [Functions - Calc] : Padding periodic boundary coefficients (N + 1 th element) for plotting.
    @in     [const MatrixType&] - Matrix to be padded for plot.
    @ps     [std::string]       - Experiment Setups.
*/
template<typename MatrixType>
MatrixType pad_plot(const MatrixType& in, Preset& ps){
    MatrixType plot = in;
    plot.conservativeResize(ps.Nx + 1, ps.Ny + 1);
    plot.row(ps.Ny) = plot.row(0);
    plot.col(ps.Nx) = plot.col(0);
    return plot;
}

/*
    [Functions - Calc] : RHS of vorticity transport equation without forcing term.
    @omg_h  [const MatrixCd&]    - Vorticity Field on Wavenumber Space.
    @mem    [Allocated& mem]     - Object to store temporary data.
    @ps     [Preset&]            - Experiment Setups.
    @fftp   [FFT2D_REAL& fftp]   - Object to carry out Real FFT for padded matrix.
    @ifftp  [IFFT2D_REAL& ifftp] - Object to carry out Real IFFT for padded matrix.
*/
MatrixCd RHS_All(const MatrixCd& omg_h, Allocated& mem, Preset& ps, FFT2D_REAL& fftp, IFFT2D_REAL& ifftp){
    // -- Linear Term Calculation
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            mem.lin_h(i, j) = - ps.nu * ps.K(i, j) * omg_h(i, j);
        }
    }

    // -- Advection Term Calculation
    // -- >> Stream Function (Poisson Equation)
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            mem.psi_h(i, j) = ps.K_inv(i, j) * omg_h(i, j);
        }
    }

    // -- >> Stream Function (Poisson Equation) == DEPRECATED CODE ==
    //mem.psi_h = omg_h.cwiseProduct(ps.K_inv);

    // -- >> omega Partial Derivatives on Wavenumber Space
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            mem.domgdx_h(i, j) = I * ps.kx(i) * omg_h(i, j);
            mem.domgdy_h(i, j) = I * ps.ky(j) * omg_h(i, j);
        }
    }

    // -- >> u_h, v_h on Wavenumber Space
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            mem.u_h(i, j) =  I * ps.ky(j) * mem.psi_h(i, j);
            mem.v_h(i, j) = -I * ps.kx(i) * mem.psi_h(i, j);
        }
    }

    // -- >> Advection term on Physical Space - Step.1
    pad<MatrixCd>(mem.u_h, mem.u_hp, ps);
    pad<MatrixCd>(mem.v_h, mem.v_hp, ps);
    pad<MatrixCd>(mem.domgdx_h, mem.domgdx_hp, ps);
    pad<MatrixCd>(mem.domgdy_h, mem.domgdy_hp, ps);
    mem.u_p        = ifftp.execute(mem.u_hp);
    mem.v_p        = ifftp.execute(mem.v_hp);
    mem.domgdx_p   = ifftp.execute(mem.domgdx_hp);
    mem.domgdy_p   = ifftp.execute(mem.domgdy_hp);
    
    // -- >> Advection term on Physical Space - Step.2
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nxp; i++){
        for(int32_t j = 0; j < ps.Nyp; j++){
            mem.adv_p(i, j) = - mem.u_p(i, j) * mem.domgdx_p(i, j) - mem.v_p(i, j) * mem.domgdy_p(i, j);
        }
    }
    // -- >> Advection term on Physical Space - Step.2 == DEPRECATED CODE ==
    //mem.adv_p     = - mem.u_p.cwiseProduct(mem.domgdx_p) - mem.v_p.cwiseProduct(mem.domgdy_p);

    // -- >> Advection term on Wavenumber Space
    mem.adv_hp     = fftp.execute(mem.adv_p);
    chop<MatrixCd>(mem.adv_hp, mem.adv_h, ps);
    mem.adv_h *= 2.25;  // 2.25 = 1.5 * 1.5
    
    return mem.lin_h + mem.adv_h;
}

/*
    [Functions - Calc] : RHS of vorticity transport equation without forcing and linear term.
    @omg_h  [const MatrixCd&]    - Vorticity Field on Wavenumber Space.
    @mem    [Allocated& mem]     - Object to store temporary data.
    @ps     [Preset&]            - Experiment Setups.
    @fftp   [FFT2D_REAL& fftp]   - Object to carry out Real FFT for padded matrix.
    @ifftp  [IFFT2D_REAL& ifftp] - Object to carry out Real IFFT for padded matrix.
*/
MatrixCd RHS_Adv(const MatrixCd& omg_h, Allocated& mem, Preset& ps, FFT2D_REAL& fftp, IFFT2D_REAL& ifftp){
    // -- Advection Term Calculation
    // -- >> Stream Function (Poisson Equation)
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            mem.psi_h(i, j) = ps.K_inv(i, j) * omg_h(i, j);
        }
    }

    // -- >> Stream Function (Poisson Equation) == DEPRECATED CODE ==
    //mem.psi_h = omg_h.cwiseProduct(ps.K_inv);

    // -- >> omega Partial Derivatives
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            mem.domgdx_h(i, j) = I * ps.kx(i) * omg_h(i, j); 
            mem.domgdy_h(i, j) = I * ps.ky(j) * omg_h(i, j);
        }
    }

    // -- >> u, v Velocities
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            mem.u_h(i, j) =  I * ps.ky(j) * mem.psi_h(i, j);
            mem.v_h(i, j) = -I * ps.kx(i) * mem.psi_h(i, j);
        }
    }

    // -- >> Advection term on Physical Space - Step.1
    pad<MatrixCd>(mem.u_h, mem.u_hp, ps);
    pad<MatrixCd>(mem.v_h, mem.v_hp, ps);
    pad<MatrixCd>(mem.domgdx_h, mem.domgdx_hp, ps);
    pad<MatrixCd>(mem.domgdy_h, mem.domgdy_hp, ps);
    mem.u_p        = ifftp.execute(mem.u_hp);
    mem.v_p        = ifftp.execute(mem.v_hp);
    mem.domgdx_p   = ifftp.execute(mem.domgdx_hp);
    mem.domgdy_p   = ifftp.execute(mem.domgdy_hp);
    
    // -- >> Advection term on Physical Space - Step.2
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nxp; i++){
        for(int32_t j = 0; j < ps.Nyp; j++){
            mem.adv_p(i, j) = - mem.u_p(i, j) * mem.domgdx_p(i, j) - mem.v_p(i, j) * mem.domgdy_p(i, j);
        }
    }
    // -- >> Advection term on Physical Space - Step.2 == DEPRECATED CODE ==
    //mem.adv_p     = - mem.u_p.cwiseProduct(mem.domgdx_p) - mem.v_p.cwiseProduct(mem.domgdy_p);

    // -- >> Advection term on Wavenumber Space
    mem.adv_hp     = fftp.execute(mem.adv_p);
    chop<MatrixCd>(mem.adv_hp, mem.adv_h, ps);
    mem.adv_h *= 2.25;  // 2.25 = 1.5 * 1.5
    
    return mem.adv_h;
}

/*
    [Functions - Math] : Calculate L2 Norm of Velocity Field with Vorticity Field on Wavenumber Space.
    @omg_h  [const MatrixCd&] - Vorticity Field on Wavenumber Space.
    @ps     [Preset&]         - Experiment Setups.
*/
double L2Norm_Velocity(MatrixCd& omg_h, Preset& ps){
    // -- Temporary Velocity Field Initialization
    MatrixCd ux_h(ps.Nx, ps.Ny);
    MatrixCd uy_h(ps.Nx, ps.Ny);
    ux_h.setZero();
    uy_h.setZero();

    // -- Calculate Velocity Field on Physical Space
    //#pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            double denom = ps.K(i, j);
            if(!(i == 0 && j == 0)){
                ux_h(i, j) = omg_h(i, j) * std::complex<double>(0.0, - ps.ky(j)) / denom;
                uy_h(i, j) = omg_h(i, j) * std::complex<double>(0.0,   ps.kx(i)) / denom;
            }
        }
    }
    // -- Calculate L2Norm
    return ps.Lx * std::sqrt((ux_h.cwiseAbs2().sum() + uy_h.cwiseAbs2().sum()));
}

/*
    [Functions - Math] : Calculate H1 Norm of Velocity Field with Vorticity Field on Wavenumber Space.
    @omg_h  [const MatrixCd&] - Vorticity Field on Wavenumber Space.
    @ps     [Preset&]         - Experiment Setups.
*/
double H1Norm_Velocity(){ return 0.0; };     // TODO

/*
    [Functions - Math] : Calculate Negative H1 Norm of Velocity Field with Vorticity Field on Wavenumber Space.
    @omg_h  [const MatrixCd&] - Vorticity Field on Wavenumber Space.
    @ps     [Preset&]         - Experiment Setups.
*/
double N_H1Norm_Velocity(){ return 0.0; };   // TODO

/*
    [Functions - Math] : Generate Taylor Vortex-Field on Wavenumber Space.
    @x0     [double]      - x center of vortex
    @y0     [double]      - y center of vortex
    @a0     [double]      - vortex spread parameter
    @Umax   [double]      - vortex max velocity parameter
    @ps     [Preset&]     - Experiment Setups.
    @fft    [FFT2D_REAL&] - Object to carry out Real FFT for matrix.
*/
MatrixCd taylor_vortex(double x0, double y0, double a0, double Umax, Preset& ps, FFT2D_REAL& fft){
    // -- Temporary Matrices
    MatrixRd omg = MatrixRd::Zero(ps.Nx, ps.Ny);
    MatrixRd X0  = x0 * MatrixRd::Ones(ps.Nx, ps.Ny);
    MatrixRd Y0  = y0 * MatrixRd::Ones(ps.Nx, ps.Ny);

    // -- Generate Taylor Vortex on Physical Space
    for(int32_t i = -1; i <= 1; i++){
        for(int32_t j = -1; j <= 1; j++){
            MatrixRd iLx = static_cast<double>(i) * ps.Lx * MatrixRd::Ones(ps.Nx, ps.Ny);
            MatrixRd jLy = static_cast<double>(j) * ps.Ly * MatrixRd::Ones(ps.Nx, ps.Ny);
            MatrixRd R2 = (ps.X_m - X0 - iLx).array().square() + (ps.Y_m - Y0 - jLy).array().square();
            MatrixRd EXP = (0.5 * (MatrixRd::Ones(ps.Nx, ps.Ny) - R2 / (a0 * a0))).array().exp();
            omg += (Umax / a0) * (2.0 * MatrixRd::Ones(ps.Nx, ps.Ny) - R2 / (a0 * a0)).cwiseProduct(EXP);
        }
    }

    // -- Transform Vortex Field into Wavenumber Space
    MatrixCd omg_h = fft.execute(omg);

    return omg_h;
}

/*
    [Functions - Math] : Generate Initial Vorticity Field on Wavenumber Space
    @type   [int32_t]     - Type of Initial Condition
    @ps     [Preset&]     - Experiment Setups.
    @fft    [FFT2D_REAL&] - Object to carry out Real FFT for matrix.

    * NOTE *
        0 : Zero Matrix Intialization
        1 : Single Taylor Vortex at the Center
        2 : Twin Taylor Vortices along x axis at the Center
        3 : A hundred of random (seeded wiht config Seed_i) Taylor Vortices
*/
MatrixCd init_omg_h(Preset& ps, FFT2D_REAL& fft){
    MatrixCd omg_h;
    // -- RNG with Mersenne Twister Engine
    std::mt19937 rnd_src(ps.seed_i);
    // -- RNG with Standard Uniform Distribution
    std::uniform_real_distribution<> dist(0.0, 1.0);
    switch (ps.type)
    {
        case 1: // Single Taylor Vortex at the Center
            {
                omg_h  = taylor_vortex(ps.Lx / 2.0, ps.Ly / 2.0, ps.Lx / 8.0, 1.0, ps, fft);
            }
            break;
        case 2: // Twin Taylor Vortices along x axis at the Center
            {
                omg_h  = taylor_vortex(ps.Lx * 0.6, ps.Ly / 2.0, ps.Lx / 10.0, 1.0, ps, fft);
                omg_h += taylor_vortex(ps.Lx * 0.4, ps.Ly / 2.0, ps.Lx / 10.0, 1.0, ps, fft);
            }   
            break;
        case 3: // A hundred of Taylor Vortices at random size, location
            {
                int32_t Nv = 100;
                omg_h  = taylor_vortex(ps.Lx * dist(rnd_src), ps.Ly * dist(rnd_src), ps.Lx / 20.0, dist(rnd_src) * 2.0 - 1.0, ps, fft);
                for(int32_t n = 1; n < Nv; n++){
                    omg_h += taylor_vortex(ps.Lx * dist(rnd_src), ps.Ly * dist(rnd_src), ps.Lx / 20.0, dist(rnd_src) * 2.0 - 1.0, ps, fft);
                }
            }
            break;
        case 0: // Zero Matrix Initialization
            {
                omg_h = MatrixCd::Zero(ps.Nx, ps.Ny);
            }
            break;
        default:
            std::cout << "Type of Initial Condition must be 0, 1, 2 or 3." << std::endl;
            std::exit(-1);
            break;
    }

    return omg_h;
}

/*
    [Functions - Math] : Restore Velocities on Physical Space from Wavenumber Space
    @omg_h  [const MatrixCd&] - Vorticity Field on Wavenumber Space.
    @ps     [Preset&]         - Experiment Setups.
    @ifft   [IFFT2D_REAL&]    - Object to carry out Real IFFT for matrix.
*/
Velocities omg_h2vel(const MatrixCd& omg_h, Preset& ps, IFFT2D_REAL& ifft){
    Velocities vels = Velocities();
    
    MatrixCd u_h  (ps.Nx, ps.Ny);
    MatrixCd v_h  (ps.Nx, ps.Ny);
    MatrixCd psi_h(ps.Nx, ps.Ny);

    // -- >> Stream Function (Poisson Equation)
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            psi_h(i, j) = ps.K_inv(i, j) * omg_h(i, j);
        }
    }
    // -- >> Stream Function (Poisson Equation) == DEPRECATED ==
    //psi_h = omg_h.cwiseProduct(ps.K_inv);

    // -- >> u_h, v_h on Wavenumber Space
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            u_h(i, j) =  I * ps.ky(j) * psi_h(i, j);
            v_h(i, j) = -I * ps.kx(i) * psi_h(i, j);
        }
    }

    // -- Inverse Transform to get u, v, psi, omg
    vels.u = ifft.execute(u_h);
    vels.v = ifft.execute(v_h);
    vels.psi = ifft.execute(psi_h);
    vels.omg = ifft.execute(omg_h);

    return vels;
}

/*
    [Functions - Math] : Calculate CFL condition of Velocity Field.
    @omg_h  [const MatrixCd&] - Vorticity Field on Wavenumber Space.
    @ps     [Preset&]         - Experiment Setups.
    @ifft   [IFFT2D_REAL&]    - Object to carry out Real IFFT for matrix.
*/
double CFL(MatrixCd& omg_h, Preset& ps, IFFT2D_REAL& inv){
    MatrixCd u_h     (ps.Nx, ps.Ny);
    MatrixCd v_h     (ps.Nx, ps.Ny);
    MatrixCd psi_h   (ps.Nx, ps.Ny);

    // -- Stream Function (Poisson Equation)
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            psi_h(i, j) = ps.K_inv(i, j) * omg_h(i, j);
        }
    }
    // -- >> Stream Function (Poisson Equation) == DEPRECATED ==
    //psi_h = omg_h.cwiseProduct(ps.K_inv);

    // -- >> u_h, v_h on Wavenumber Space
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            u_h(i, j) =  I * ps.ky(j) * psi_h(i, j);
            v_h(i, j) = -I * ps.kx(i) * psi_h(i, j);
        }
    }

    // -- Inverse Transform to get u, v
    MatrixRd u = inv.execute(u_h);
    MatrixRd v = inv.execute(v_h);
    MatrixRd combined = u.cwiseAbs() + v.cwiseAbs();

    return ps.Nx * ps.dt / (2.0 * ps.Lx) * combined.maxCoeff();
}

/*
    [Functions - Math] : Calculate Energy Spectra of Velocity Field.
    @omg_h  [const MatrixCd&] - Vorticity Field on Wavenumber Space.
    @ps     [Preset&]         - Experiment Setups.
*/
MatrixRd EnergySpectra(MatrixCd& omg_h, Preset& ps){

    MatrixCd ux_h(ps.Nx, ps.Ny);
    MatrixCd uy_h(ps.Nx, ps.Ny);
    ux_h.setZero();
    uy_h.setZero();

    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            double denom = ps.K(i, j);
            if(!(i == 0 && j == 0)){
                ux_h(i, j) = omg_h(i, j) * std::complex<double>(0.0, - ps.ky(j)) / denom;
                uy_h(i, j) = omg_h(i, j) * std::complex<double>(0.0,   ps.kx(i)) / denom;
            }
        }
    }

    // -- k = [0, 1, 2, ..., max_coeff, max_coeff + 1] but 0 and max_coeff + 1 never gonna be used.
    int32_t max_r = std::round(std::sqrt(ps.K.maxCoeff())) + 1;
    MatrixRd es(max_r, 2);
    es.setZero();

    // -- Calculate Energy Spectra
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            int32_t rnd_k = std::round(std::sqrt(ps.K(i, j)));
            es(rnd_k, 0) = static_cast<double>(rnd_k);
            double tmp = (ps.Lx * ps.Ly) * (std::abs(ux_h(i, j)) * std::abs(ux_h(i, j)) + std::abs(uy_h(i, j)) * std::abs(uy_h(i, j)));
            es(rnd_k, 1) += tmp;
        }
    }

    return es;
}

/*
    [Functions - Main] : Main Function.
    @argc   [int32_t] - Runtime arguments count.
    @argv   [char**]  - Runtime arguments array.
*/
int32_t main(int32_t argc, char** argv){

    const std::string CONFIG_FILE_PATH = "./config.json";

    /*
        >> Load Configuration (Proc Load-1)
    */
    // -- Load Json Configuration
    std::cout << ">> CONFIGURATION      || Attempting to load Configuration." << std::endl;
    std::ifstream ifs(CONFIG_FILE_PATH);
    if(!ifs.is_open()){
        std::cout << "-- Unable to Read Config File = " << CONFIG_FILE_PATH << std::endl;
    }
    nlohmann::json jsonobj = nlohmann::json::parse(ifs);
    ifs.close();
    // -- Output Json Configuration
    std::cout << ">> CONFIGURATION      || Initializing with setups below." << std::endl;
    std::cout << jsonobj.dump(4) << std::endl;
    // -- Deserialize Json Object
    auto config = jsonobj.template get<Setup::JsonConfig>();

    /*
        >> Directory Preparation (Proc Load-2)
    */
    std::cout << ">> FILE SYSTEM        || Initializing directory." << std::endl;
    CreateDir(config.RootDir);
    std::string out_init    = config.RootDir + "/init/";
    CreateDir(out_init);
    std::string out_mid     = config.RootDir + "/mid/";
    CreateDir(out_mid);
    std::string out_final   = config.RootDir + "/final/";
    CreateDir(out_final);

    /*
        >> Threads Preparation (Proc Load-3)
    */
    std::cout << ">> PRALALLELIZATION   || Initializing with setups below." << std::endl;
    std::cout << " -- Available Threads = " << omp_get_max_threads() << std::endl;
    std::cout << " -- OMP, FFTW Threads = " << config.Threads << std::endl;
    fftw_init_threads();
    fftw_plan_with_nthreads(config.Threads);

    /*
        >> Presets Preparation (Proc Load-4)
    */
    Preset ps = Preset();

    // -- Domain Size               (FIXED)
    ps.Lx           = 2.0 * PI;
    ps.Ly           = 2.0 * PI;

    // -- Grid Size                 (POWER OF 2 SUGGESTED BUT AT LEAST EVEN)
    ps.Nx           = config.N;
    ps.Ny           = config.N;
    // -- Grid Size Padded
    ps.Nxp          = ps.Nx * 3 / 2;
    ps.Nyp          = ps.Ny * 3 / 2;

    // -- Temporal Discretization
    ps.dt           = config.Delta_t;
    ps.t_max        = config.Max_t;
    ps.t_out        = config.Out_t;
    ps.t_info       = config.Info_t;

    // -- Physical params
    ps.nu           = config.nu;

    // -- Kolmogorov Forcing
    ps.gamma        = config.gamma;
    ps.k            = config.k;

    // -- Seed for Initial Condition
    ps.type         = config.Type;
    ps.seed_i       = config.Seed_i;
    
    // -- Wavenumbers
    ps.kx           = VectorRd::Zero(ps.Nx);
    ps.ky           = VectorRd::Zero(ps.Ny);
    for(int32_t i = 0; i < ps.Nx; i++){
        if(i <= ps.Nx / 2){ // 0, 1, 2, ... , Nx/2
            ps.kx(i) = static_cast<double>(i) / ps.Lx * 2.0 * PI;
        }
        else{               // -Nx/2 + 1, ... , -1
            ps.kx(i) =  - static_cast<double>(ps.Nx - i) / ps.Lx * 2.0 * PI;
        }
    }
    for(int32_t j = 0; j < ps.Ny; j++){
        if(j <= ps.Ny / 2){ // 0, 1, 2, ... , Ny/2
            ps.ky(j) = static_cast<double>(j) / ps.Ly * 2.0 * PI;
        }
        else{               // -Ny/2 + 1, ... , -1
            ps.ky(j) =  - static_cast<double>(ps.Ny - j) / ps.Ly * 2.0 * PI;
        }
    }

    // -- Wavenumbers Matrix [(kx(i)^2 + ky(j)^2)] -- pre calc
    ps.K            = MatrixRd::Zero(ps.Nx, ps.Ny);
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            ps.K(i, j) = ps.kx(i) * ps.kx(i) + ps.ky(j) * ps.ky(j);
        }
    }

    // -- Wavenumbers Matrix cwise Inverse [1 / (kx(i)^2 + ky(j)^2)] and K_inv(0, 0) = 0.0 -- pre calc
    ps.K_inv        = MatrixRd::Zero(ps.Nx, ps.Ny);
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            if(i == 0 && j == 0){
                ps.K_inv(i, j) = 0.0;
            }
            else{
                ps.K_inv(i, j) = 1.0 / (ps.kx(i) * ps.kx(i) + ps.ky(j) * ps.ky(j));
            }
        }
    }

    // -- Mesh Grids
    ps.X            = VectorRd::LinSpaced(ps.Nx, 0.0, ps.Lx);
    ps.Y            = VectorRd::LinSpaced(ps.Ny, 0.0, ps.Ly);
    ps.X_m          = MatrixRd::Zero(ps.Nx, ps.Ny);
    ps.Y_m          = MatrixRd::Zero(ps.Nx, ps.Ny);
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            ps.X_m(i, j) = ps.X(i);
            ps.Y_m(i, j) = ps.Y(j);
        }
    }

    /*
        >> Initialize FFT, IFFT (Proc Init-1)
    */
    FFT2D_REAL  fwd_r  =  FFT2D_REAL(ps.Nx,  ps.Ny,  FFTW_ESTIMATE);  
    FFT2D_REAL  fwdp_r =  FFT2D_REAL(ps.Nxp, ps.Nyp, FFTW_ESTIMATE);
    IFFT2D_REAL inv_r  = IFFT2D_REAL(ps.Nx,  ps.Ny,  FFTW_ESTIMATE);
    IFFT2D_REAL invp_r = IFFT2D_REAL(ps.Nxp, ps.Nyp, FFTW_ESTIMATE);

    /*
        >> Initial Condition Generation (Proc Init-2)
    */
    std::cout << ">> Initial Condition      || " << std::endl;
    // -- Omg_h Generation (using intial condition type)
    std::cout << "-- Omg_h ic integer = " << ps.type << std::endl;
    MatrixCd omg_h = init_omg_h(ps, fwd_r);

    // -- g_h   Generation (using kolmogorov forcing)
    MatrixCd g_h = MatrixCd::Zero(ps.Nx, ps.Ny);
    // kx(0) = 0,  ky(ps.k)         =  ps.k
    g_h(0, ps.k)         = std::complex<double>(- config.gamma * static_cast<double>(ps.k) / 2, 0);
    // kx(0) = 0,  ky(ps.Ny - ps.k) = -ps.k
    g_h(0, ps.Ny - ps.k) = std::complex<double>(- config.gamma * static_cast<double>(ps.k) / 2, 0);

    /*
        >> Output Initial Condition (Proc Init-3)
    */
    Velocities vels_init            = omg_h2vel(omg_h, ps, inv_r);
    std::string out_init_omg_bin    = out_init + "omg_init.dat";
    std::string out_init_omg_txt    = out_init + "omg_init.txt";
    std::string out_init_omg_h_bin  = out_init + "omg_h_init.dat";
    std::string out_init_omg_h_txt  = out_init + "omg_h_init.txt";
    std::string out_g_bin           = out_init + "g.dat";
    std::string out_g_txt           = out_init + "g.txt";
    std::string out_g_h_bin         = out_init + "g_h.dat";
    std::string out_g_h_txt         = out_init + "g_h.txt";
    WriteBinary (pad_plot(vels_init.omg, ps),       out_init_omg_bin);
    WriteText   (pad_plot(vels_init.omg, ps),       out_init_omg_txt, IOFMT);
    WriteBinary (omg_h,                             out_init_omg_h_bin);
    WriteText   (omg_h,                             out_init_omg_h_txt, IOFMT);
    WriteBinary (pad_plot(inv_r.execute(g_h), ps),  out_g_bin);
    WriteText   (pad_plot(inv_r.execute(g_h), ps),  out_g_txt, IOFMT);
    WriteBinary (g_h,                               out_g_h_bin);
    WriteText   (g_h,                               out_g_h_txt, IOFMT);

/*
        >> Memory Allocation for Calculation -- RHS (Proc Init-4a)
    */
    Allocated mem       = Allocated();
    // fourier coeff - linear term hat
    mem.lin_h           = MatrixCd::Zero(ps.Nx, ps.Ny);
    // fourier coeff - advection term hat
    mem.adv_h           = MatrixCd::Zero(ps.Nx, ps.Ny);
    // fourier coeff - stream function hat
    mem.psi_h           = MatrixCd::Zero(ps.Nx, ps.Ny);
    // fourier coeff - x dir velocity hat
    mem.u_h             = MatrixCd::Zero(ps.Nx, ps.Ny);
    // fourier coeff - y dir velocity hat
    mem.v_h             = MatrixCd::Zero(ps.Nx, ps.Ny);
    // fourier coeff - d omg / d x hat
    mem.domgdx_h        = MatrixCd::Zero(ps.Nx, ps.Ny);
    // fourier coeff - d omg / d y hat
    mem.domgdy_h        = MatrixCd::Zero(ps.Nx, ps.Ny);
    // fourier coeff - advection term hat padded
    mem.adv_hp          = MatrixCd::Zero(ps.Nxp, ps.Nyp);
    // fourier coeff - x dir velocity hat padded
    mem.u_hp            = MatrixCd::Zero(ps.Nxp, ps.Nyp);
    // fourier coeff - y dir velocity hat padded
    mem.v_hp            = MatrixCd::Zero(ps.Nxp, ps.Nyp);
    // fourier coeff - d omg / d x hat padded
    mem.domgdx_hp       = MatrixCd::Zero(ps.Nxp, ps.Nyp);
    // fourier coeff - d omg / d y hat padded
    mem.domgdy_hp       = MatrixCd::Zero(ps.Nxp, ps.Nyp);

    // real space value - x dir velocity padded
    mem.u_p             = MatrixRd::Zero(ps.Nxp, ps.Nyp);
    // real space value - y dir velocity padded
    mem.v_p             = MatrixRd::Zero(ps.Nxp, ps.Nyp);
    // real space value - d omg / d x padded
    mem.domgdx_p        = MatrixRd::Zero(ps.Nxp, ps.Nyp);
    // real space value - d omg / d y padded
    mem.domgdy_p        = MatrixRd::Zero(ps.Nxp, ps.Nyp);
    // real space value - adcection term padded
    mem.adv_p           = MatrixRd::Zero(ps.Nxp, ps.Nyp);

    /*
        >> Memory Allocation for Calculation -- RK4 (Proc Init-4b)
    */
    // -- RK4 Steps
    MatrixCd k1 = MatrixCd::Zero(ps.Nx, ps.Ny);
    MatrixCd k2 = MatrixCd::Zero(ps.Nx, ps.Ny);
    MatrixCd k3 = MatrixCd::Zero(ps.Nx, ps.Ny);
    MatrixCd k4 = MatrixCd::Zero(ps.Nx, ps.Ny);

    // -- Exponential of ExpRK4 
    MatrixRd Lambda_Dt_half  = MatrixRd::Zero(ps.Nx, ps.Ny);
    MatrixRd Lambda_Dt_one   = MatrixRd::Zero(ps.Nx, ps.Ny);
    MatrixRd Lambda_Dt_two   = MatrixRd::Zero(ps.Nx, ps.Ny);
    MatrixRd Lambda_Dt_three = MatrixRd::Zero(ps.Nx, ps.Ny);
    #pragma omp parallel for collapse(2)
    for(int32_t i = 0; i < ps.Nx; i++){
        for(int32_t j = 0; j < ps.Ny; j++){
            Lambda_Dt_half(i, j)  = std::exp(- ps.nu * ps.K(i, j) * ps.dt * 0.5);
            Lambda_Dt_one(i, j)   = std::exp(- ps.nu * ps.K(i, j) * ps.dt * 1.0);
            Lambda_Dt_two(i, j)   = std::exp(- ps.nu * ps.K(i, j) * ps.dt * 2.0);
            Lambda_Dt_three(i, j) = std::exp(- ps.nu * ps.K(i, j) * ps.dt * 3.0);
        }
    }

    /*
        >> Memory Allocation for data -- Norms and CFL Condition (Proc Init-5)
    */
    MatrixRd mat_L2Norm_Vel = MatrixRd(static_cast<int>(ps.c_max() + 1), 2);
    MatrixRd mat_CFL_Vel    = MatrixRd(static_cast<int>(ps.c_max() + 1), 2);

    /*
        >> Main Loop (Proc Main-1)
    */
    std::cout << ">> MAIN LOOP          || Started Calculation." << std::endl;
    for(int32_t c = 0; c <= ps.c_max(); c++){

        // -- Condition Check
        double L2Norm_Vel_c = L2Norm_Velocity(omg_h, ps);
        double CFL_Vel_c = CFL(omg_h, ps, inv_r);

        mat_L2Norm_Vel(c, 0) = c * ps.dt;
        mat_L2Norm_Vel(c, 1) = L2Norm_Vel_c;
        mat_CFL_Vel(c, 0)    = c * ps.dt;
        mat_CFL_Vel(c, 1)    = CFL_Vel_c;

        if(CFL_Vel_c > 1.0){
            std::cout << " [c = " << c << "]" << std::endl;
            std::cout << " -- CFL condition violated!" << std::endl;
            std::cout << " -- CFL = " << CFL_Vel_c << std::endl;
            std::exit(0);
        }

        // -- Output Information
        if(c % ps.c_info() == 0){
            std::cout << " [c = " << c << "]" << std::endl;
            std::cout << " -- |u| = " << L2Norm_Vel_c << std::endl;
            std::cout << " -- CFL = " << CFL_Vel_c << std::endl;
        }

        // -- Output Data -mid-
        if(c % ps.c_out() == 0){
            Velocities vels = omg_h2vel(omg_h, ps, inv_r);
            std::string out_mid_omg_bin     = out_mid + "omg_" + std::to_string(c) + ".dat";
            std::string out_mid_omg_txt     = out_mid + "omg_" + std::to_string(c) + ".txt";
            std::string out_mid_omg_h_bin   = out_mid + "omg_h_" + std::to_string(c) + ".dat";
            std::string out_mid_omg_h_txt   = out_mid + "omg_h_" + std::to_string(c) + ".txt";
            std::string out_mid_u_bin       = out_mid + "u_" + std::to_string(c) + ".dat";
            std::string out_mid_u_txt       = out_mid + "u_" + std::to_string(c) + ".txt";
            std::string out_mid_v_bin       = out_mid + "v_" + std::to_string(c) + ".dat";
            std::string out_mid_v_txt       = out_mid + "v_" + std::to_string(c) + ".txt";
            std::string out_mid_es_bin      = out_mid + "energy_spectra_" + std::to_string(c) + ".dat";
            std::string out_mid_es_txt      = out_mid + "energy_spectra_" + std::to_string(c) + ".txt";
            WriteBinary (pad_plot(vels.omg, ps),    out_mid_omg_bin);
            WriteText   (pad_plot(vels.omg, ps),    out_mid_omg_txt, IOFMT);
            WriteBinary (omg_h,                     out_mid_omg_h_bin);
            WriteText   (omg_h,                     out_mid_omg_h_txt, IOFMT);

            WriteBinary (pad_plot(vels.u, ps),      out_mid_u_bin);
            WriteText   (pad_plot(vels.u, ps),      out_mid_u_txt, IOFMT);
            WriteBinary (pad_plot(vels.v, ps),      out_mid_v_bin);
            WriteText   (pad_plot(vels.v, ps),      out_mid_v_txt, IOFMT);

            WriteBinary (EnergySpectra(omg_h, ps),  out_mid_es_bin);
            WriteText   (EnergySpectra(omg_h, ps),  out_mid_es_txt, IOFMT);
        }

        // -- ExpRK4 Calculation
        k1 = RHS_Adv(omg_h                   , mem, ps, fwdp_r, invp_r) + g_h;
        k2 = RHS_Adv(omg_h + 0.5 * ps.dt * k1, mem, ps, fwdp_r, invp_r) + g_h;
        k3 = RHS_Adv(omg_h + 0.5 * ps.dt * k2, mem, ps, fwdp_r, invp_r) + g_h;
        k4 = RHS_Adv(omg_h +       ps.dt * k3, mem, ps, fwdp_r, invp_r) + g_h;
        #pragma omp parallel for collapse(2)
        for(int32_t i = 0; i < ps.Nx; i++){
            for(int32_t j = 0; j < ps.Ny; j++){
                omg_h(i, j) = Lambda_Dt_one(i, j) * omg_h(i, j) + ps.dt / 6.0 * (Lambda_Dt_one(i, j) * k1(i, j) + 2.0 * Lambda_Dt_half(i, j) * (k2(i, j) + k3(i, j)) + k4(i, j));
            }
        }
    }

    /*
        >> Output Results (Proc Main-2)
    */
    std::string out_final_u_L2_norm_bin  = out_final + "u_L2_norm.dat";
    std::string out_final_u_L2_norm_txt  = out_final + "u_L2_norm.txt";
    std::string out_final_cfl_bin        = out_final + "cfl.dat";
    std::string out_final_cfl_txt        = out_final + "cfl.txt";
    WriteBinary (mat_L2Norm_Vel, out_final_u_L2_norm_bin);
    WriteText   (mat_L2Norm_Vel, out_final_u_L2_norm_txt, IOFMT);
    WriteBinary (mat_CFL_Vel, out_final_cfl_bin);
    WriteText   (mat_CFL_Vel, out_final_cfl_txt, IOFMT);

    std::cout << "Max CFL Value = " << mat_CFL_Vel.col(1).maxCoeff() << std::endl;

    /*
        >> Threads Finalizing (Proc Load-5)
    */
    std::cout << ">> PRALALLELIZATION   || Finalizing." << std::endl;
    fftw_cleanup_threads();

    return 0;
}