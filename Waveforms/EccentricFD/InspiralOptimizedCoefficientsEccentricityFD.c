/*
 *  Copyright (C) 2014 Eliu Huerta
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 */

#include "InspiralEccentricFD.h"

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif


static double UNUSED
C0(double eta)
{
    return ( 3./(128.*eta));
}

static double UNUSED
C1(double Mtotal)
{
    return(PI*Mtotal);
}

/*case 1*/

static double UNUSED
C2(double e0, double f0)
{
    double e2,e4,e6,e8,p1;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p1= pow(f0, 19./9.);
    return(p1*((-2355.*e2)/1462. - (2608555.*e4)/444448. - (1326481225.*e6)/1.01334144e8 - (6505217202575.*e8)/2.77250217984e11)  );
}

static double UNUSED
C3(double e0, double f0)
{
    double e2,e4,e6,e8,p2;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p2= pow(f0, 19./3.);
    return( p2*((-75356125.*e6)/3.326976e6 - (250408403375.*e8)/1.011400704e9) );
}

static double UNUSED
C4(double e0, double f0)
{
    double e2,e4,e6,e8,p3;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p3= pow(f0, 38./9.);
    return(  p3*((5222765.*e4)/998944. + (17355248095.*e6)/4.55518464e8 +
                 (128274289063885.*e8)/8.30865678336e11) );
}

static double UNUSED
C5(double e0, double f0)
{
    double e2,e4,e6,e8,p4;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p4= pow(f0, 76./9.);
    return(  (4537813337273.*e8*p4)/(39444627456.) );
}

/*case 2*/

static double UNUSED
C6(double eta)
{
    return(  3715./756. + (55.*eta)/9.);
}

static double UNUSED
C7(double eta, double e0, double f0)
{
    double e2,e4,e6,e8,p1;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p1= pow(f0, 19./9.);
    return(p1*((-583255.*e2)/122808. - (1938156365.*e4)/1.12000896e8 -(985575550175*e6)/2.5536204288e10 - (690482340216175.*e8)/9.981007847424e12 - (8635.*e2*eta)/1462. - (28694105.*e4*eta)/1.333344e6 -   (14591293475.*e6*eta)/3.04002432e8 - (71557389228325.*e8*eta)/8.31750653952e11)  );

}

static double UNUSED
C8(double eta,double e0, double f0)
{
    double e2,e4,e6,e8,p2;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p2= pow(f0, 19./3.);
    return(p2*((-31207858707289299381995.*e6)/5.3807139303898e20 -(103703714484322341846369385.*e8)/1.635737034838499e23 -(462027517873731215615.*e6*eta)/6.405611821892619e18 -(1535317441894408829488645.*e8*eta)/1.9473059938553564e21) );

}

static double UNUSED
C9(double eta,double e0, double f0)
{
    double e2,e4,e6,e8,p3;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p3= pow(f0, 38./9.);
    return(p3*((1867778296655755.*e4)/1.34516772125568e14 +(6206627279787073865.*e6)/6.133964808925901e16 +(45873772442848006154795.*e8)/1.1188351811480843e20 +(27652168591135.*e4*eta)/1.601390144352e12 +(91888156228341605.*e6*eta)/7.30233905824512e14 +(679154100768947601215.*e8*eta)/1.33194664422391e18 ) );
}


static double UNUSED
C10(double eta,double e0, double f0)
{
    double e2,e4,e6,e8,p4;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p4= pow(f0, 76./9.);
    return(p4*((2222952136716664456903429135345831.*e8)/7.66183921683643e30 +
               (32910462320165961003953863376587.*e8*eta)/9.121237162900513e28) );
}

/*case 3*/

static double UNUSED
C11(double UNUSED eta)
{
    return( -16.*PI);
}


static double UNUSED
C12(double UNUSED eta, double e0, double f0)
{
    double e2,e4,e6,e8,p1;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p1=pow(f0, 19./9.);
    return(PI*p1*((7536.*e2)/731. + (521711.*e4)/13889. + (265296245.*e6)/3.166692e6 + (1301043440515.*e8)/8.664069312e9 ) );
}


static double UNUSED
C13(double UNUSED eta, double e0, double f0)
{
    double e2,e4,e6,e8,p2;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p2=pow(f0, 19./3.);
    return(PI*p2*((7800202806596945705.*e6)/6.672512314471478e16 +(25920073926321650577715.*e8)/2.0284437435993293e19) );
}

static double UNUSED
C14(double UNUSED eta, double e0, double f0)
{
    double e2,e4,e6,e8,p3;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p3=pow(f0, 38./9.);
    return(PI*p3*((-475065859669.*e4)/1.6681147337e10 - (1578643851680087.*e6)/7.606603185672e12 -(11667906828579178421.*e8)/1.3874444210665728e16) );
}

static double UNUSED
C15(double UNUSED eta, double e0, double f0)
{
    double e2,e4,e6,e8,p4;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p4=pow(f0, 76./9.);
    return(-((2759558801317902042155334469457.*e8*p4*PI)/4750644355677350183868572160.));
}

/*case 4*/

static double UNUSED
C16(double eta)
{
    return(15293365./508032. + (27145.*eta)/504. + (3085.*eta*eta)/72.);
}

static double UNUSED
C17(double eta,double e0, double f0)
{
    double e2,e4,e6,e8,p1;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p1=pow(f0, 19./9.);
    return(p1*((-2401058305.*e2)/2.47580928e8 - (7978716747515.*e4)/2.25793806336e11 -(4057272307914425.*e6)/5.1480987844608e13 -(2842476030950240425.*e8)/2.0121711820406784e16 - (4261765.*e2*eta)/245616. -(14161845095.*e4*eta)/2.24001792e8 - (7201466570525.*e6*eta)/5.1072408576e10 -(5045260598968525.*e8*eta)/1.9962015694848e13 -(484345.*e2*eta*eta)/35088. - (1609478435.*e4*eta*eta)/3.2000256e7 -(818438915825.*e6*eta*eta)/7.296058368e9 -(4013719013988775.*e8*eta*eta)/1.9962015694848e13) );
}

static double UNUSED
C18(double eta,double e0, double f0)
{
    double e2,e4,e6,e8,p2;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p2=pow(f0, 19./3.);
    return(p2*( (-110474729572613560552840685.*e6)/1.0847519283665837e24 -(367107526369794861717089596255.*e8)/3.297645862234414e26 -(196087423156943883913505.*e6*eta)/1.07614278607796e21 -(651598507150524526244577115.*e8*eta)/3.271474069676998e23 -(22285124348468295519365.*e6*eta*eta)/1.5373468372542285e20 -(74053468209960146010849895.*e8*eta*eta)/4.673534385252855e22 ) );
}

static double UNUSED
C19(double eta,double e0, double f0)
{
    double e2,e4,e6,e8,p3;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p3=pow(f0, 38./9.);
    return(p3*((6841716503626986565.*e4)/2.711858126051451e17 +(22735023941552476355495.*e6)/1.2366073054794616e20 +(168036723934429498871218085.*e8)/2.255571725194538e23 +(12143723404950745.*e4*eta)/2.69033544251136e14 +(40353592874651325635.*e6*eta)/1.2267929617851802e17 +(298257242353143912203705.*e8*eta)/2.2376703622961686e20 +(1380121079545885.*e4*eta*eta)/3.8433363464448e13 +(4586142347330975855.*e6*eta*eta)/1.7525613739788288e16 +(33896614207384379043965.*e8*eta*eta)/3.1966719461373837e19 ) );
}


static double UNUSED
C20(double eta,double e0, double f0)
{
    double e2,e4,e6,e8,p4;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p4=pow(f0, 76./9.);
    return(p4*( (7770499311332403684925585429305516241.*e8)/1.5446267861142245e34 +(13792268987637324946295665896844693.*e8*eta)/1.532367843367286e31 +(1567476508633676458254637292015689.*e8*eta*eta)/2.189096919096123e30 ));
}

/*case 5*/

static double UNUSED
C21(double eta)
{
    return( PI*(38645./756. -65.*eta/9.));
}

static double UNUSED
C22(double eta,double e0, double f0)
{
    double e2,e4,e6,e8,p1;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p1=pow(f0, 19./9.);
    return(PI*p1*( (6067265.*e2)/122808. + (20161521595.*e4)/1.12000896e8 +(10252373388025.*e6)/2.5536204288e10 +(7182689108386025.*e8)/9.981007847424e12 - (10205.*e2*eta)/1462. -(33911215.*e4*eta)/1.333344e6 - (17244255925.*e6*eta)/3.04002432e8 -(84567823633475.*e8*eta)/8.31750653952e11) );
}

static double UNUSED
C23(double eta,double e0, double f0)
{
    double e2,e4,e6,e8,p2;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p2=pow(f0, 19./3.);
    return(PI*p2*((36828693183369973116475.*e6)/7.686734186271143e19 +(122381747448338420666046425.*e8)/2.3367671926264273e22 -(433615096349678814025.*e6*eta)/6.405611821892619e18 -(1440902965169982699005075.*e8*eta)/1.9473059938553564e21 ) );
}

static double UNUSED
C24(double eta,double e0, double f0)
{
    double e2,e4,e6,e8,p3;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p3=pow(f0, 38./9.);
    return( PI*p3*( (-2316846009950855.*e4)/1.9216681732224e13 -(7698879291066691165.*e6)/8.762806869894144e15 -(56903148963613058870695.*e8)/1.5983359730686919e19 +(27278171420045.*e4*eta)/1.601390144352e12 +(90645363628809535.*e6*eta)/7.30233905824512e14 +(669968502482700007405.*e8*eta)/1.33194664422391e18));
}

static double UNUSED
C25(double eta,double e0, double f0)
{
    double e2,e4,e6,e8,p4;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p4=pow(f0, 76./9.);
    return(PI*p4*( (-18041156334765213110425249225007393.*e8)/7.66183921683643e30 + (2334216112662079584736091244017.*e8*eta)/7.016336279154241e27));
}

/*case 6*/

static double UNUSED
C26(double eta)
{
    return( 11583231236531./4694215680.- (15737765635.*eta)/3.048192e6 +
           (76055.*eta*eta)/1728. - (127825.*eta*eta*eta)/1296. -
           (6848.*GAMMA)/21. -(640.*PI*PI)/3. +(2255.*eta*PI*PI)/12.);
}

/* The following cases should be multiplied by log(4.*pow(PI*Mtotal*f, 1./3.)) */
static double UNUSED
C27(double UNUSED eta)
{
    return( -6848./21. );/* this term is multiplied by log(4x)*/
}

static double UNUSED
C28(double UNUSED eta,double e0, double f0)
{
    double e2,e4,e6,e8,p1;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p1=pow(f0, 19./9.);
    return(p1*((-537568.*e2)/5117. - (111646154.*e4)/291669. - (28386698215.*e6)/3.3250266e7 -(19887378305015.*e8)/1.2996103968e10) );/* this term is multiplied by log(4x)*/
}


static double UNUSED
C29(double UNUSED eta, double e0, double f0)
{
    double e2,e4,e6,e8,p2;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p2=pow(f0, 19./3.);
    return(p2*( (-657204451373573967571.*e6)/7.006137930195053e17 -(2183890391914386294238433.*e8)/2.1298659307792957e20) );/* this term is multiplied by log(4x)*/
}

static double UNUSED
C30(double UNUSED eta,double e0, double f0)
{
    double e2,e4,e6,e8,p3;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p3=pow(f0, 38./9.);
    return(p3*((83880153412870.*e4)/3.50304094077e11 +(139366874895483505.*e6)/7.9869333449556e13 + (1030073825416757818915.*e8)/1.4568166421199014e17 ) );/* this term is multiplied by log(4x)*/
}

static double UNUSED
C31(double UNUSED eta,double e0, double f0)
{
    double e2,e4,e6,e8,p4;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p4=pow(f0, 76./9.);
    return((229018097854706671787019720252139.*e8*p4)/49881765734612176930620007680.);/* this term is multiplied by log(4x)*/
}

static double UNUSED
C32(double eta, double e0, double f0)
{
    double e2,e4,e6,e8,p1;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p1=pow(f0, 19./9.);
    return(p1*( (1578237767500487.*e2)/2.28764777472e12 +(5244484101404118301.*e4)/2.08633477054464e15 +(533376501191162085059.*e6)/9.513686553683558e16 +(373677141943502503806739.*e8)/3.718492344411174e19 -(2470829204695.*e2*eta)/1.485485568e9 -(8210565447201485.*e4*eta)/1.354762838016e12 -(4175170127655540575.*e6*eta)/3.08885927067648e14 -(2925073821111304814575.*e8*eta)/1.207302709224407e17 +(11940635.*e2*eta*eta)/842112. +(39678730105.*e4*eta*eta)/7.68006144e8 +(20177105913475.*e6*eta*eta)/1.75105400832e11 +(98950858868368325.*e8*eta*eta)/4.79088376676352e14 -(20068525.*e2*eta*eta*eta)/631584. -(66687708575.*e4*eta*eta*eta)/5.76004608e8 -(33911492517125.*e6*eta*eta*eta)/1.31329050624e11 -(166305877783829875.*e8*eta*eta*eta)/3.59316282507264e14 -(537568.*e2*GAMMA)/5117. - (111646154.*e4*GAMMA)/291669. -(28386698215.*e6*GAMMA)/3.3250266e7 -(19887378305015.*e8*GAMMA)/1.2996103968e10 -(50240.*e2*PI*PI)/731. -(10434220.*e4*PI*PI)/41667. -(1326481225.*e6*PI*PI)/2.375019e6 -(6505217202575.*e8*PI*PI)/6.498051984e9 +(354035.*e2*eta*PI*PI)/5848. +(1176458305.*e4*eta*PI*PI)/5.333376e6 +(598243032475.*e6*eta*PI*PI)/1.216009728e9 +(2933852958361325.*e8*eta*PI*PI)/3.327002615808e12 ) );
}

static double UNUSED
C33(double eta, double e0, double f0)
{
    double e2,e4,e6,e8,p2;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p2=pow(f0, 19./3.);
    return(p2*( (62534662812190508908451288422363.*e6)/1.0023107818107234e28 +(207802684524909061102783631427512249.*e8)/3.047024776704599e30 -(96662893738280943308126058155.*e6*eta)/6.508511570199502e24 -(321210795892307574612902891249065.*e8*eta)/1.9785875173406487e27 +(8813910165617557415555.*e6*eta*eta)/6.961570583792733e19 +(29288623480347143291889265.*e8*eta*eta)/2.116317457472991e22 -(785113635484365349577225.*e6*eta*eta*eta)/2.7672243070576115e21 -(2608932610714546056645118675.*e8*eta*eta*eta)/8.412361893455138e23 -(657204451373573967571.*e6*GAMMA)/7.006137930195053e17 -(2183890391914386294238433.*e8*GAMMA)/2.1298659307792957e20 -(30710488381942708765.*e6*PI*PI)/5.004384235853609e16 - (102050952893195621226095.*e8*PI*PI)/1.5213328076994972e19 +(13850430260256161653015.*e6*eta*PI*PI)/2.5622447287570477e19 + (46024979754831225172968845.*e8*eta*PI*PI)/7.789223975421425e21) );
}

static double UNUSED
C34(double eta, double e0, double f0)
{
    double e2,e4,e6,e8,p3;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p3=pow(f0, 38./9.);
    return(p3*( (-796520595220381729758415.*e4)/5.0115138169430816e20 -(2646837937917328487987213045.*e6)/2.285250300526045e23 -(19563030899655064495340095274735.*e8)/4.168296548159506e26 +(6168627083362586227675.*e4*eta)/1.6271148756308705e18 +(20498347798013874034564025.*e6*eta)/7.41964383287677e20 +(151505237861278885566710654075.*e8*eta)/1.3533430351167228e24 -(562467383866675.*e4*eta*eta)/1.7403787229184e13 -(1869079116588961025.*e6*eta*eta)/7.936126976507904e15 -(13814541490402312805075.*e8*eta*eta)/1.4475495605150417e19 +(50102713130841625.*e4*eta*eta*eta)/6.91800542360064e14 +(166491315733786719875.*e6*eta*eta*eta)/3.154610473161892e17 +(1230553147045766992549625.*e8*eta*eta*eta)/5.754009503047291e20 +(83880153412870.*e4*GAMMA)/3.50304094077e11 +(139366874895483505.*e6*GAMMA)/7.9869333449556e13 +(1030073825416757818915.*e8*GAMMA)/1.4568166421199014e17 +(7839266674100.*e4*PI*PI)/5.0043442011e10 +(6512470789508575.*e6*PI*PI)/5.704952389254e12 +(48134290907325131725.*e8*PI*PI)/1.0405833157999296e16 - (883877317504775.*e4*eta*PI*PI)/6.405560577408e12 -(2937124326068367325.*e6*eta*PI*PI)/2.920935623298048e15 - (21708565199203634407975.*e8*eta*PI*PI)/5.32778657689564e18) );
}

static double UNUSED
C35(double eta, double e0, double f0)
{
    double e2,e4,e6,e8,p4;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p4=pow(f0, 76./9.);
    return(p4*( (-21803333020968369933785155127082363071942707.*e8)/7.136175751847717e38 + (6736884392917513798607444845703601288679.*e8*eta)/9.267760716685346e34 -(614282187703745932737146352451999.*e8*eta*eta)/9.91289170911452e29 + (54718202538837159478833263067719005.*e8*eta*eta*eta)/3.940374454373021e31 +(229018097854706671787019720252139.*e8*GAMMA)/4.988176573461218e28 +(2140356054716884783056259067777.*e8*PI*PI)/7.125966533516026e26 - (965300580677315037158372839567427.*e8*eta*PI*PI)/3.648494865160205e29));
}

/* case 7*/

static double UNUSED
C36(double eta)
{
    return( PI*(77096675./254016. + (378515.*eta)/1512. - (74045.*eta*eta)/756.));
}

static double UNUSED
C37(double eta, double e0, double f0)
{
    double e2,e4,e6,e8,p1;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p1=pow(f0, 19./9.);
    return(PI*p1*((12104177975.*e2)/6.1895232e7 + (40222183410925.*e4)/5.6448451584e10 +(20453458379485375.*e6)/1.2870246961152e13 +(14329446184895255375.*e8)/5.030427955101696e15 + (59426855*e2*eta)/368424. +(197475439165.*e4*eta)/3.36002688e8 +(100418608176175.*e6*eta)/7.6608612864e10 +(70352065412362175.*e8*eta)/2.9943023542272e13 -(11625065.*e2*eta*eta)/184212. -(38630090995.*e4*eta*eta)/1.68001344e8 -(19643860461025.*e6*eta*eta)/3.8304306432e10 -(13762251650419025.*e8*eta*eta)/1.4971511771136e13 ) );

}


static double UNUSED
C38(double eta, double e0, double f0)
{
    double e2,e4,e6,e8,p2;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p2=pow(f0, 19./3.);
    return(PI*p2*((434593322826290843757317275.*e6)/2.7118798209164593e23 +(1444153611751764473805565304825.*e8)/8.244114655586036e25 +(2133685941573919740699595.*e6*eta)/1.61421417911694e21 +(7090238383850135298344754185.*e8*eta)/4.907211104515498e23 -(417391055952448085809285.*e6*eta*eta)/8.0710708955847e20 -(1386990478929984989144254055.*e8*eta*eta)/2.453605552257749e23 ) );
}

static double UNUSED
C39(double eta, double e0, double f0)
{
    double e2,e4,e6,e8,p3;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p3=pow(f0, 38./9.);
    return(PI*p3*( (-28083426938595740975.*e4)/6.779645315128627e16 -(93321227716953647259925.*e6)/3.091518263698654e19 -(689746068418917003152253775.*e8)/5.638929312986345e22 -(137878817052260255.*e4*eta)/4.03550316376704e14 -(458171309064660827365.*e6*eta)/1.8401894426777702e17 -(3386387715003096689295295.*e8*eta)/3.3565055434442526e20 +(26971816199185265.*e4*eta*eta)/2.01775158188352e14 +(89627345229892635595.*e6*eta*eta)/9.200947213388851e16 +(662444231688055412226385.*e8*eta*eta)/1.6782527717221263e20) );
}

static double UNUSED
C40(double eta, double e0, double f0)
{
    double e2,e4,e6,e8,p4;
    e2=e0*e0;
    e4=e2*e2;
    e6=e4*e2;
    e8=e6*e2;
    p4=pow(f0, 76./9.);
    return(PI*p4*((-1776389932392373211125413523490219335.*e8)/2.2715099795797417e32 -(8721390841557034022662273046715703.*e8*eta)/1.3520892735593702e30 +(1706076073241722479183197515934809.*e8*eta*eta)/6.760446367796851e29 ) );
}

           /* The following coefficients are for the real part of zeta_l's*/

    /*case 1 of zeta_re*/
static double UNUSED
z1(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,si,ci,P,Q,p5, s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p5=pow(f0, 133./18.);
    return( p5*( (46031066168471.*c2b*e7*P)/(1.2136808448e11) + (46031066168471.*c2b*c2i*e7*P)/(1.2136808448e11) - (46031066168471.*ci*e7*Q*s2b)/(6.068404224e10) - (8391437082143.*e7*P*s2i)/(3.6410425344e10)));
}

static double UNUSED
z2(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,si,ci,P,Q,p6, s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p6=pow(f0, 95./18.);
    return(p6*( -(717415013.*c2b*e5*P)/(1.3307904e7) -(717415013.*c2b*c2i*e5*P)/(1.3307904e7) -(11919850440995.*c2b*e7*P)/(2.4273616896e10) -(11919850440995.*c2b*c2i*e7*P)/(2.4273616896e10) +(717415013.*ci*e5*Q*s2b)/(6.653952e6) +(11919850440995.*ci*e7*Q*s2b)/(1.2136808448e10)+(220389695.*e5*P*s2i)/(6.653952e6) +(3661774782425.*e7*P*s2i)/(1.2136808448e10)) );
}

static double UNUSED
z3(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,si,ci,P,Q,p7,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p7=pow(f0, 19./6.);
    return(p7*( (30299.*c2b*e3*P)/(3648.) +(30299.*c2b*c2i*e3*P)/(3648.) +(100683577.*c2b*e5*P)/(2.217984e6) +(100683577.*c2b*c2i*e5*P)/(2.217984e6) +(384584085937.*c2b*e7*P)/(2.697068544e9) +(384584085937.*c2b*c2i*e7*P)/(2.697068544e9)-(30299.*ci*e3*Q*s2b)/(1824.) -(100683577.*ci*e5*Q*s2b)/(1.108992e6) -(384584085937.*ci*e7*Q*s2b)/(1.348534272e9)-(9517.*e3*P*s2i)/(1824.) -(31624991.*e5*P*s2i)/(1.108992e6) -(120798928871.*e7*P*s2i)/(1.348534272e9)) );
}

static double UNUSED
z4(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,si,ci,P,Q,p8,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p8=pow(f0, 19./18.);
    return(p8*(-(3.*c2b*e0*P)/(2.) -(3.*c2b*c2i*e0*P)/(2.) -(3323.*c2b*e3*P)/(1216.) -(3323.*c2b*c2i*e3*P)/(1216.) -(15994231.*c2b*e5*P)/(4.435968e6) -(15994231.*c2b*c2i*e5*P)/(4.435968e6) -(105734339801.*c2b*e7*P)/(2.4273616896e10) -(105734339801.*c2b*c2i*e7*P)/(2.4273616896e10)+3.*ci*e0*Q*s2b +(3323.*ci*e3*Q*s2b)/(608.) +(15994231.*ci*e5*Q*s2b)/(2.217984e6) +(105734339801.*ci*e7*Q*s2b)/(1.2136808448e10)+e0*P*s2i +(3323.*e3*P*s2i)/(1824.) +(15994231.*e5*P*s2i)/(6.653952e6) +(105734339801.*e7*P*s2i)/(3.6410425344e10)) );
}

    /* case 2 of zeta_real*/

static double UNUSED
z5(double UNUSED f0, double inc, double bet, double FPlus, double FCross)
{
    double c2b,s2b,ci,P,Q,c2i;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    return(2.*c2b*P + 2.*c2b*c2i*P - 4.*ci*Q*s2b);
}

static double UNUSED
z6(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,si,ci,P,Q,p1,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p1=pow(f0, 19./9.);
    return(p1*(-(277.*c2b*e2*P)/(24.) -(277.*c2b*c2i*e2*P)/(24.) -(920471.*c2b*e4*P)/(21888.) -(920471.*c2b*c2i*e4*P)/(21888.) -(468070445.*c2b*e6*P)/(4.990464e6) -(468070445.*c2b*c2i*e6*P)/(4.990464e6) -(2295471547915.*c2b*e8*P)/(1.3653909504e10) -(2295471547915.*c2b*c2i*e8*P)/(1.3653909504e10)+e2*P*s2i +(3323.*e4*P*s2i)/(912.) +(1689785.*e6*P*s2i)/(207936.) +(8286900895.*e8*P*s2i)/(5.68912896e8)+(277.*ci*e2*Q*s2b)/(12.) +(920471.*ci*e4*Q*s2b)/(10944.) +(468070445.*ci*e6*Q*s2b)/(2.495232e6) +(2295471547915.*ci*e8*Q*s2b)/(6.826954752e9)));
}

static double UNUSED
z7(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,si,ci,P,Q,p2,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p2=pow(f0, 19./3.);
    return(p2*(-(103729904239.*c2b*e6*P)/(1.9961856e8) -(103729904239.*c2b*c2i*e6*P)/(1.9961856e8) -(344694471786197.*c2b*e8*P)/(6.068404224e10) -(344694471786197.*c2b*c2i*e8*P)/(6.068404224e10)+(103729904239.*ci*e6*Q*s2b)/(9.980928e7) +(344694471786197.*ci*e8*Q*s2b)/(3.034202112e10)+(29064841.*e6*P*s2i)/(554496.) +(96582466643.*e8*P*s2i)/(1.68566784e8)) );
}

static double UNUSED
z8(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,si,ci,P,Q,p3,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p3 = pow (f0, 38./9.);
    return(p3*((3254599.*c2b*e4*P)/(43776.) +(3254599.*c2b*c2i*e4*P)/(43776.) +(10815032477.*c2b*e6*P)/(1.9961856e7) +(10815032477.*c2b*c2i*e6*P)/(1.9961856e7) +(79934933490791.*c2b*e8*P)/(3.6410425344e10) +(79934933490791.*c2b*c2i*e8*P)/(3.6410425344e10)-(3254599.*ci*e4*Q*s2b)/(21888.) -(10815032477.*ci*e6*Q*s2b)/(9.980928e6) -(79934933490791.*ci*e8*Q*s2b)/(1.8205212672e10)-(3305.*e4*P*s2i)/(456.) -(10982515.*e6*P*s2i)/(207936.) -(81172812745.*e8*P*s2i)/(3.79275264e8)) );
}

static double UNUSED
z9(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,si,ci,P,Q,p4,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p4 = pow (f0, 76./9.);
    return(p4*((8340362777769439.*c2b*e8*P)/(2.18462552064e12) +(8340362777769439.*c2b*c2i*e8*P)/(2.18462552064e12)-(8340362777769439.*ci*e8*Q*s2b)/(1.09231276032e12)-(4442498396267.*e8*P*s2i)/(1.137825792e10)) );
}

    /* case 3 of zeta_real*/

static double UNUSED
z10(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,si,ci,P,Q,p5,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p5 = pow (f0, 133./18.);
    return(p5*((-74006050878931.*c2b*e7*P)/(4.045602816e10) -(74006050878931.*c2b*c2i*e7*P)/(4.045602816e10)+(74006050878931.*ci*e7*Q*s2b)/(2.022801408e10)+(2531702819.*e7*P*s2i)/(2.957312e7)) );
}

static double UNUSED
z11(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,si,ci,P,Q,p6,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p6 = pow (f0, 95./18.);
    return(p6*((1810486747.*c2b*e5*P)/(7.39328e6) +(1810486747.*c2b*c2i*e5*P)/(7.39328e6) +(6016247460281.*c2b*e7*P)/(2.697068544e9) +(6016247460281.*c2b*c2i*e7*P)/(2.697068544e9)-(1810486747.*ci*e5*Q*s2b)/(3.69664e6) -(6016247460281.*ci*e7*Q*s2b)/(1.348534272e9)-(50883.*e5*P*s2i)/(4864.) -(281807015.*e7*P*s2i)/(2.957312e6)) );
}

static double UNUSED
z12(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,si,ci,P,Q,p7,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p7 = pow (f0, 19./6.);
    return(p7*(-(40863.*c2b*e3*P)/(1216.) -(40863.*c2b*c2i*e3*P)/(1216.) -(135787749.*c2b*e5*P)/(739328.) -(135787749.*c2b*c2i*e5*P)/(739328.) -(518672547069.*c2b*e7*P)/(8.99022848e8) -(518672547069.*c2b*c2i*e7*P)/(8.99022848e8)+(40863.*ci*e3*Q*s2b)/(608.) +(135787749.*ci*e5*Q*s2b)/(369664.) +(518672547069.*ci*e7*Q*s2b)/(4.49511424e8)+(9.*e3*P*s2i)/(8.) +(29907.*e5*P*s2i)/(4864.) +(114236667.*e7*P*s2i)/(5.914624e6)) );
}

static double UNUSED
z13(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p8,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p8 = pow (f0, 19./18.);
    return(p8*((9.*c2b*e0*P)/(2.) +(9.*c2b*c2i*e0*P)/(2.) +(9969.*c2b*e3*P)/(1216.) +(9969.*c2b*c2i*e3*P)/(1216.) +(15994231.*c2b*e5*P)/(1.478656e6) +(15994231.*c2b*c2i*e5*P)/(1.478656e6) +(105734339801.*c2b*e7*P)/(8.091205632e9) +(105734339801.*c2b*c2i*e7*P)/(8.091205632e9)-9.*ci*e0*Q*s2b -(9969.*ci*e3*Q*s2b)/(608.) -(15994231.*ci*e5*Q*s2b)/(739328.) -(105734339801.*ci*e7*Q*s2b)/(4.045602816e9)) );
}

    /*case 4 for z_real*/

static double UNUSED
z14(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p1,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p1 = pow (f0, 19./9.);
    return(p1*(8.*c2b*e2*P +8.*c2b*c2i*e2*P +(3323.*c2b*e4*P)/(114.) +(3323.*c2b*c2i*e4*P)/(114.) +(1689785.*c2b*e6*P)/(25992.) +(1689785.*c2b*c2i*e6*P)/(25992.) +(8286900895.*c2b*e8*P)/(7.1114112e7) +(8286900895.*c2b*c2i*e8*P)/(7.1114112e7)-16.*ci*e2*Q*s2b -(3323.*ci*e4*Q*s2b)/(57.) -(1689785.*ci*e6*Q*s2b)/(12996.) -(8286900895.*ci*e8*Q*s2b)/(3.5557056e7) ) );
}

static double UNUSED
z15(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,si,ci,P,Q,p2,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p2 = pow (f0, 19./3.);
    return(p2*((71474367.*c2b*e6*P)/(115520.) +(71474367.*c2b*c2i*e6*P)/(115520.) +(237509321541.*c2b*e8*P)/(3.511808e7) +(237509321541.*c2b*c2i*e8*P)/(3.511808e7)-(71474367.*ci*e6*Q*s2b)/(57760.) -(237509321541.*ci*e8*Q*s2b)/(1.755904e7)-(51793.*e6*P*s2i)/(3420.) -(172108139.*e8*P*s2i)/(1.03968e6)) );
}

static double UNUSED
z16(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,si,ci,P,Q,p3,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p3 = pow (f0, 38./9.);
    return(p3*(-(1431.*c2b*e4*P)/(19.) -(1431.*c2b*c2i*e4*P)/(19.) -(1585071.*c2b*e6*P)/(2888.) -(1585071.*c2b*c2i*e6*P)/(2888.) -(3905136831.*c2b*e8*P)/(1.755904e6) -(3905136831.*c2b*c2i*e8*P)/(1.755904e6)+(2862.*ci*e4*Q*s2b)/(19.) +(1585071.*ci*e6*Q*s2b)/(1444.) +(3905136831.*ci*e8*Q*s2b)/(877952.)+(4.*e4*P*s2i)/(3.) +(3323.*e6*P*s2i)/(342.) +(24560609.*e8*P*s2i)/(623808.)) );
}

static double UNUSED
z17(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,si,ci,P,Q,p4,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p4 = pow (f0, 76./9.);
    return(p4*((-49436673769121.*c2b*e8*P)/(9.95597568e9) -(49436673769121.*c2b*c2i*e8*P)/(9.95597568e9)+(865873231.*e8*P*s2i)/(6.23808e6)+(49436673769121.*ci*e8*Q*s2b)/(4.97798784e9)) );
}

                        /*case 5 for zeta_real*/

static double UNUSED
z18(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,si,ci,P,Q,p5,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p5 = pow (f0, 133./18.);
    return(p5*((3053741715625.*c2b*e7*P)/(2.235727872e9) +(3053741715625.*c2b*c2i*e7*P)/(2.235727872e9)-(3053741715625.*ci*e7*Q*s2b)/(1.117863936e9)-(15300625.*e7*P*s2i)/(700416.)) );
}

static double UNUSED
z19(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,si,ci,P,Q,p6,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p6 = pow (f0, 95./18.);
    return(p6*(-(13023125.*c2b*e5*P)/(87552.) -(13023125.*c2b*c2i*e5*P)/(87552.) -(216379221875.*c2b*e7*P)/(1.59694848e8) -(216379221875.*c2b*c2i*e7*P)/(1.59694848e8)+(13023125.*ci*e5*Q*s2b)/(43776.) +(216379221875.*ci*e7*Q*s2b)/(7.9847424e7)+(625.*e5*P*s2i)/(384.) + (10384375.*e7*P*s2i)/(700416.)) );
}

static double UNUSED
z20(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p7,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p7 = pow (f0, 19./6.);
    return(p7*((625.*c2b*e3*P)/(48.) +(625.*c2b*c2i*e3*P)/(48.) +(2076875.*c2b*e5*P)/(29184.) +(2076875.*c2b*c2i*e5*P)/(29184.) +(7933101875.*c2b*e7*P)/(3.5487744e7) +(7933101875.*c2b*c2i*e7*P)/(3.5487744e7)-(625.*ci*e3*Q*s2b)/(24.) -(2076875.*ci*e5*Q*s2b)/(14592.) -(7933101875.*ci*e7*Q*s2b)/(1.7743872e7)) );
}

/* case 6 for zeta_real*/

static double UNUSED
z21(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,si,ci,P,Q,p2,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p2 = pow (f0, 19./3.);
    return(p2*(-(1656963.*c2b*e6*P)/(6080.) -(1656963.*c2b*c2i*e6*P)/(6080.) -(5506088049.*c2b*e8*P)/(1.84832e6) -(5506088049.*c2b*c2i*e8*P)/(1.84832e6)+(1656963.*ci*e6*Q*s2b)/(3040.) +(5506088049.*ci*e8*Q*s2b)/(924160.)+(81.*e6*P*s2i)/(40.) +(269163.*e8*P*s2i)/(12160.)) );
}

static double UNUSED
z22(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p3,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p3 = pow (f0, 38./9.);
    return(p3*((81.*c2b*e4*P)/(4.) +(81.*c2b*c2i*e4*P)/(4.) +(89721.*c2b*e6*P)/(608.) +(89721.*c2b*c2i*e6*P)/(608.) +(221045481.*c2b*e8*P)/(369664.) +(221045481.*c2b*c2i*e8*P)/(369664.)-(81.*ci*e4*Q*s2b)/(2.) -(89721.*ci*e6*Q*s2b)/(304.) -(221045481.*ci*e8*Q*s2b)/(184832.)) );
}

static double UNUSED
z23(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,si,ci,P,Q,p4,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p4 = pow (f0, 76./9.);
    return(p4*( (71730555921.*c2b*e8*P)/(2.587648e7) +(71730555921.*c2b*c2i*e8*P)/(2.587648e7)-(333693.*e8*P*s2i)/(10640.)-(71730555921.*ci*e8*Q*s2b)/(1.293824e7)) );
}

    /* case 7 for zeta_real*/

static double UNUSED
z24(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,si,ci,P,Q,p5,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p5 = pow (f0, 133./18.);
    return(p5*((-1109077123.*c2b*e7*P)/(2.33472e6) -(1109077123.*c2b*c2i*e7*P)/(2.33472e6)+(1109077123.*ci*e7*Q*s2b)/(1.16736e6)+(117649.*e7*P*s2i)/(46080.) ) );
}

static double UNUSED
z25(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p6,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p6 = pow (f0, 95./18.);
    return(p6*((117649.*c2b*e5*P)/(3840.) +(117649.*c2b*c2i*e5*P)/(3840.) +(390947627.*c2b*e7*P)/(1.400832e6) +(390947627.*c2b*c2i*e7*P)/(1.400832e6)-(117649.*ci*e5*Q*s2b)/(1920.) -(390947627.*ci*e7*Q*s2b)/(700416.)) );
}

    /* case 8 for zeta_real*/

static double UNUSED
z26(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p2,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p2 = pow (f0, 19./3.);
    return(p2*((2048.*c2b*e6*P)/(45.) +(2048.*c2b*c2i*e6*P)/(45.) +(425344.*c2b*e8*P)/(855.) +(425344.*c2b*c2i*e8*P)/(855.)-(4096.*ci*e6*Q*s2b)/(45.) -(850688.*ci*e8*Q*s2b)/(855.)) );
}


static double UNUSED
z27(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,si,ci,P,Q,p4,s2i,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    si=sin(inc);
    ci=cos(inc);
    s2i=si*si;
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p4 = pow (f0, 76./9.);
    return(p4*((-14348288.*c2b*e8*P)/(17955.) -(14348288.*c2b*c2i*e8*P)/(17955.)+(28696576.*ci*e8*Q*s2b)/(17955.)+(1024.*e8*P*s2i)/(315.)) );
}

        /* case 9 for zeta_real*/

static double UNUSED
z28(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p5,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p5 = pow (f0, 133./18.);
    return(p5*((4782969.*c2b*e7*P)/(71680.) +(4782969.*c2b*c2i*e7*P)/(71680.) -(4782969.*ci*e7*Q*s2b)/(35840.)) );
}

        /*case 10 for zeta_real*/

static double UNUSED
z29(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p4,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p4 = pow (f0, 76./9.);
    return((390625.*e8*p4*(c2b*P + c2b*c2i*P - 2.*ci*Q*s2b))/(4032.) );
}

    /*The following are the coefficients for the imaginary part of the zeta_l's*/

                            /*case 1 for zeta_im*/

static double UNUSED
q1(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p5,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p5 = pow (f0, 133./18.);
    return(p5*((16099508821573.*c2b*ci*e7*Q)/(2.022801408e10)+(16099508821573.*e7*P*s2b)/(4.045602816e10) +(16099508821573.*c2i*e7*P*s2b)/(4.045602816e10) ) );
}

static double UNUSED
q2(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p6,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p6 = pow (f0, 95./18.);
    return(p6*(-(749695861.*c2b*ci*e5*Q)/(6.653952e6) -(12456196730515.*c2b*ci*e7*Q)/(1.2136808448e10)-(749695861.*e5*P*s2b)/(1.3307904e7) -(749695861.*c2i*e5*P*s2b)/(1.3307904e7) -(12456196730515.*e7*P*s2b)/(2.4273616896e10) -(12456196730515.*c2i*e7*P*s2b)/(2.4273616896e10) ) );
}

static double UNUSED
q3(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p7,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p7 = pow (f0, 19./6.);
    return(p7*((31363.*c2b*ci*e3*Q)/(1824.) +(104219249.*c2b*ci*e5*Q)/(1.108992e6) +(398089398569.*c2b*ci*e7*Q)/(1.348534272e9)+(31363.*e3*P*s2b)/(3648.) +(31363.*c2i*e3*P*s2b)/(3648.) +(104219249.*e5*P*s2b)/(2.217984e6) +(104219249.*c2i*e5*P*s2b)/(2.217984e6) +(398089398569.*e7*P*s2b)/(2.697068544e9) +(398089398569.*c2i*e7*P*s2b)/(2.697068544e9) ) );
}

static double UNUSED
q4(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p8,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p8 = pow (f0, 19./18.);
    return(p8*(-3.*c2b*ci*e0*Q -(3323.*c2b*ci*e3*Q)/(608.) -(15994231.*c2b*ci*e5*Q)/(2.217984e6) -(105734339801.*c2b*ci*e7*Q)/(1.2136808448e10)-(3.*e0*P*s2b)/2. -(3.*c2i*e0*P*s2b)/2. -(3323.*e3*P*s2b)/1216. -(3323.*c2i*e3*P*s2b)/(1216.) -(15994231.*e5*P*s2b)/(4.435968e6) -(15994231.*c2i*e5*P*s2b)/(4.435968e6) -(105734339801.*e7*P*s2b)/(2.4273616896e10) -(105734339801.*c2i*e7*P*s2b)/(2.4273616896e10)) );
}

    /* case 2 for zeta_im */

static double UNUSED
q5(double UNUSED f0, double inc, double bet, double FPlus, double FCross)
{
    double c2b,s2b,ci,c2i,P,Q;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    return(4.*c2b*ci*Q + 2.*P*s2b + 2.*c2i*P*s2b);
}

static double UNUSED
q6(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p1,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p1 = pow (f0, 19./9.);
    return(p1*(-(277.*c2b*ci*e2*Q)/(12.) -(920471.*c2b*ci*e4*Q)/(10944.) -(468070445.*c2b*ci*e6*Q)/(2.495232e6) -(2295471547915.*c2b*ci*e8*Q)/(6.826954752e9)-(277.*e2*P*s2b)/(24.) -(277.*c2i*e2*P*s2b)/(24.) -(920471.*e4*P*s2b)/(21888.) -(920471.*c2i*e4*P*s2b)/(21888.) -(468070445.*e6*P*s2b)/(4.990464e6) -(468070445.*c2i*e6*P*s2b)/(4.990464e6) -(2295471547915.*e8*P*s2b)/(1.3653909504e10) -(2295471547915.*c2i*e8*P*s2b)/(1.3653909504e10) ) );
}

static double UNUSED
q7(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p2,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p2 = pow (f0, 19./3.);
    return(p2*(-(104238504751.*c2b*ci*e6*Q)/(9.980928e7) -(346384551287573.*c2b*ci*e8*Q)/(3.034202112e10)-(104238504751.*e6*P*s2b)/(1.9961856e8) -(104238504751.*c2i*e6*P*s2b)/(1.9961856e8) -(346384551287573.*e8*P*s2b)/(6.068404224e10) -(346384551287573.*c2i*e8*P*s2b)/(6.068404224e10)) );
}

static double UNUSED
q8(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p3,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p3 = pow (f0, 38./9.);
    return(p3*((3265543.*c2b*ci*e4*Q)/(21888.) +(10851399389.*c2b*ci*e6*Q)/(9.980928e6) +(80203724795687.*c2b*ci*e8*Q)/(1.8205212672e10)+(3265543.*e4*P*s2b)/(43776.) +(3265543.*c2i*e4*P*s2b)/(43776.) +(10851399389.*e6*P*s2b)/(1.9961856e7) +(10851399389.*c2i*e6*P*s2b)/(1.9961856e7) +(80203724795687.*e8*P*s2b)/(3.6410425344e10) +(80203724795687.*c2i*e8*P*s2b)/(3.6410425344e10)) );
}

static double UNUSED
q9(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p4,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p4 = pow (f0, 76./9.);
    return(p4*((8388155956587871.*e8*P*s2b)/(2.18462552064e12) +(8388155956587871.*c2i*e8*P*s2b)/(2.18462552064e12)+ (8388155956587871.*c2b*ci*e8*Q)/(1.09231276032e12)) );
}

            /*case 3 for zeta_im */

static double UNUSED
q10(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p5,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p5 = pow (f0, 133./18.);
    return(p5*((-74123294656819.*c2b*ci*e7*Q)/(2.022801408e10)-(74123294656819.*e7*P*s2b)/(4.045602816e10) -(74123294656819.*c2i*e7*P*s2b)/(4.045602816e10)) );
}

static double UNUSED
q11(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p6,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p6 = pow (f0, 95./18.);
    return(p6*( (1812254203.*c2b*ci*e5*Q)/(3.69664e6) +(6022120716569.*c2b*ci*e7*Q)/(1.348534272e9)+(1812254203.*e5*P*s2b)/(7.39328e6) +(1812254203.*c2i*e5*P*s2b)/(7.39328e6) +(6022120716569.*e7*P*s2b)/(2.697068544e9) +(6022120716569.*c2i*e7*P*s2b)/(2.697068544e9)) );
}

static double UNUSED
q12(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p7,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p7 = pow (f0, 19./6.);
    return(p7*( -(40863.*c2b*ci*e3*Q)/(608.) -(135787749.*c2b*ci*e5*Q)/(369664.) -(518672547069.*c2b*ci*e7*Q)/(4.49511424e8)-(40863.*e3*P*s2b)/(1216.) -(40863.*c2i*e3*P*s2b)/(1216.) -(135787749.*e5*P*s2b)/(739328.) -(135787749.*c2i*e5*P*s2b)/(739328.) -(518672547069.*e7*P*s2b)/(8.99022848e8) -(518672547069.*c2i*e7*P*s2b)/(8.99022848e8)) );
}

static double UNUSED
q13(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p8,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p8 = pow (f0, 19./18.);
    return(p8*(9.*c2b*ci*e0*Q +(9969.*c2b*ci*e3*Q)/(608.) +(15994231.*c2b*ci*e5*Q)/(739328.) +(105734339801.*c2b*ci*e7*Q)/(4.045602816e9)+(9.*e0*P*s2b)/(2.) +(9.*c2i*e0*P*s2b)/(2.) +(9969.*e3*P*s2b)/(1216.) +(9969.*c2i*e3*P*s2b)/(1216.) +(15994231.*e5*P*s2b)/(1.478656e6) +(15994231.*c2i*e5*P*s2b)/(1.478656e6) +(105734339801.*e7*P*s2b)/(8.091205632e9) +(105734339801.*c2i*e7*P*s2b)/(8.091205632e9) ) );
}

                /* case 4 for zeta_im*/


static double UNUSED
q14(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p1,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p1 = pow (f0, 19./9.);
    return(p1*( 16.*c2b*ci*e2*Q +(3323.*c2b*ci*e4*Q)/(57.) +(1689785.*c2b*ci*e6*Q)/(12996.) +(8286900895.*c2b*ci*e8*Q)/(3.5557056e7)+8.*e2*P*s2b +8.*c2i*e2*P*s2b +(3323.*e4*P*s2b)/(114.) +(3323.*c2i*e4*P*s2b)/(114.) +(1689785.*e6*P*s2b)/(25992.) +(1689785.*c2i*e6*P*s2b)/(25992.) +(8286900895.*e8*P*s2b)/(7.1114112e7) +(8286900895.*c2i*e8*P*s2b)/(7.1114112e7)) );
}

static double UNUSED
q15(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p2,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p2 = pow (f0, 19./3.);
    return(p2*((643523447.*c2b*ci*e6*Q)/(519840.) +(2138428414381.*c2b*ci*e8*Q)/(1.5803136e8)+(643523447.*e6*P*s2b)/(1.03968e6) +(643523447.*c2i*e6*P*s2b)/(1.03968e6) +(2138428414381.*e8*P*s2b)/(3.1606272e8) +(2138428414381.*c2i*e8*P*s2b)/(3.1606272e8) ) );
}

static double UNUSED
q16(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p3,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p3 = pow (f0, 38./9.);
    return(p3*(-(2862.*c2b*ci*e4*Q)/(19.) -(1585071.*c2b*ci*e6*Q)/(1444.) -(3905136831.*c2b*ci*e8*Q)/(877952.)-(1431.*e4*P*s2b)/(19.) -(1431.*c2i*e4*P*s2b)/(19.) -(1585071.*e6*P*s2b)/(2888.) -(1585071.*c2i*e6*P*s2b)/(2888.) -(3905136831.*e8*P*s2b)/(1.755904e6) -(3905136831.*c2i*e8*P*s2b)/(1.755904e6) ) );
}

static double UNUSED
q17(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p4,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p4 = pow (f0, 76./9.);
    return(p4*( (-49470967683233.*c2b*ci*e8*Q)/(4.97798784e9)-(49470967683233.*e8*P*s2b)/(9.95597568e9) -(49470967683233.*c2i*e8*P*s2b)/(9.95597568e9)) );
}

                /* case 5 for zeta_im*/

static double UNUSED
q18(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p5,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p5 = pow (f0, 133./18.);
    return(p5*( (3054326535625.*c2b*ci*e7*Q)/(1.117863936e9)+(3054326535625.*e7*P*s2b)/(2.235727872e9) +(3054326535625.*c2i*e7*P*s2b)/(2.235727872e9)) );
}

static double UNUSED
q19(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p6,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p6 = pow (f0, 95./18.);
    return(p6*( -(13023125.*c2b*ci*e5*Q)/(43776.) -(216379221875.*c2b*ci*e7*Q)/(7.9847424e7)-(13023125.*c2i*e5*P*s2b)/(87552.) -(216379221875.*e7*P*s2b)/(1.59694848e8) -(216379221875.*c2i*e7*P*s2b)/(1.59694848e8)-(13023125.*e5*P*s2b)/(87552.)) );
}

static double UNUSED
q20(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p7,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p7 = pow (f0, 19./6.);
    return(p7*( (625.*c2b*ci*e3*Q)/(24.) +(2076875.*c2b*ci*e5*Q)/(14592.) +(7933101875.*c2b*ci*e7*Q)/(1.7743872e7)+(625.*e3*P*s2b)/(48.) +(625.*c2i*e3*P*s2b)/(48.) +(2076875.*e5*P*s2b)/(29184.) +(2076875.*c2i*e5*P*s2b)/(29184.) +(7933101875.*e7*P*s2b)/(3.5487744e7) +(7933101875.*c2i*e7*P*s2b)/(3.5487744e7)) );
}


            /*case 6 for zeta_im */

static double UNUSED
q21(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p2,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p2 = pow (f0, 19./3.);
    return(p2*( -(1656963.*c2b*ci*e6*Q)/(3040.) -(5506088049.*c2b*ci*e8*Q)/(924160.)-(1656963.*e6*P*s2b)/(6080.) -(1656963.*c2i*e6*P*s2b)/(6080.) -(5506088049.*e8*P*s2b)/(1.84832e6) -(5506088049.*c2i*e8*P*s2b)/(1.84832e6)) );
}

static double UNUSED
q22(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p3,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p3 = pow (f0, 38./9.);
    return(p3*( (81.*c2b*ci*e4*Q)/(2.) +(89721.*c2b*ci*e6*Q)/(304.) +(221045481.*c2b*ci*e8*Q)/(184832.)+(81.*e4*P*s2b)/(4.) +(81.*c2i*e4*P*s2b)/(4.) +(89721.*e6*P*s2b)/(608.) +(89721.*c2i*e6*P*s2b)/(608.) +(221045481.*e8*P*s2b)/(369664.) +(221045481.*c2i*e8*P*s2b)/(369664.)) );
}

static double UNUSED
q23(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p4,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p4 = pow (f0, 76./9.);
    return(p4*((71738041617.*c2b*ci*e8*Q)/(1.293824e7)+(71738041617.*e8*P*s2b)/(2.587648e7) +(71738041617.*c2i*e8*P*s2b)/(2.587648e7) ) );
}

                        /*case 7 for zeta_im*/


static double UNUSED
q24(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p5,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p5 = pow (f0, 133./18.);
    return(p5*((-1109077123.*c2b*ci*e7*Q)/(1.16736e6)-(1109077123.*e7*P*s2b)/(2.33472e6) -(1109077123.*c2i*e7*P*s2b)/(2.33472e6) ) );
}

static double UNUSED
q25(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p6,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p6 = pow (f0, 95./18.);
    return(p6*((117649.*c2b*ci*e5*Q)/(1920.) +(390947627.*c2b*ci*e7*Q)/(700416.)+(117649.*e5*P*s2b)/(3840.) +(117649.*c2i*e5*P*s2b)/(3840.) +(390947627.*e7*P*s2b)/(1.400832e6) +(390947627.*c2i*e7*P*s2b)/(1.400832e6) ) );
}


                    /* case 8 for zeta_im*/

static double UNUSED
q26(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p2,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p2 = pow (f0, 19./3.);
    return(p2*((4096.*c2b*ci*e6*Q)/(45.) +(850688.*c2b*ci*e8*Q)/(855.)+(2048.*e6*P*s2b)/(45.) +(2048.*c2i*e6*P*s2b)/(45.) +(425344.*e8*P*s2b)/(855.) +(425344.*c2i*e8*P*s2b)/(855.)) );
}

static double UNUSED
q27(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p4,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p4 = pow (f0, 76./9.);
    return(p4*( (-28696576.*c2b*ci*e8*Q)/(17955.)-(14348288.*e8*P*s2b)/(17955.) -(14348288.*c2i*e8*P*s2b)/(17955.)) );
}


                /*case 9 of zeta_im*/

static double UNUSED
q28(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,c2b,s2b,ci,P,Q,p5,c2i;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    c2i=ci*ci;
    P=FPlus;
    Q=FCross;
    p5 = pow (f0, 133./18.);
    return(p5*((4782969.*c2b*ci*e7*Q)/(35840.) +(4782969.*e7*P*s2b)/(71680.) +(4782969.*c2i*e7*P*s2b)/(71680.)));
}


                /*case 10 for zeta_im*/

static double UNUSED
q29(double e0, double f0, double inc, double bet, double FPlus, double FCross)
{
    double e2,e3,e4,e5,e6,e7,e8,c2b,s2b,ci,P,Q,p4;
    e2=e0*e0;
    e3=e2*e0;
    e4=e3*e0;
    e5=e4*e0;
    e6=e5*e0;
    e7=e6*e0;
    e8=e7*e0;
    c2b=cos(2.*bet);
    s2b=sin(2.*bet);
    ci=cos(inc);
    P=FPlus;
    Q=FCross;
    p4 = pow (f0, 76./9.);
    return(p4*(390625.*e8*(2.*c2b*ci*Q + P*s2b + ci*ci*P*s2b))/(4032.));
}
