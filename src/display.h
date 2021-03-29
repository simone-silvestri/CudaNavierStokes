
#ifndef DISPLAY_H_
#define DISPLAY_H_

#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "math.h"
#include <ctime>

#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <algorithm>
using namespace std;

#ifdef LINUX
  #include "GL/gl.h"
  #include "GL/glu.h"
  #include "GL/glut.h"
#else
  #include "/usr/include/GL/gl.h"
  #include "/usr/include/GL/glu.h"
  #include "/usr/include/GL/glut.h"
#endif

#include "globals.h"

int width  = 3000;
int height = 3000;
float aspectRatio = float(height)/float(width);

GLdouble clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop;

int jd = 0;
int kd = 0;

void initDisplay(int argc, char** argv);
void display();
void reshape(GLsizei width, GLsizei height);
void key(unsigned char c, int x, int y);



void initDisplay(int argc, char** argv)  {


        if(argc>1) {
	  jd=strtol(argv[1], nullptr, 0);
	}
	if(argc>2) {
          kd=strtol(argv[2], nullptr, 0);	
        }

        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
        glutCreateWindow("Graph plotter");
	glutInitWindowSize(width, height);   			// Set the window's initial width & height - non-square
	glutInitWindowPosition(150, 150); 			// Position the window's initial top-left corner

        /* Register GLUT callbacks. */
        glutDisplayFunc(display);
      	glutKeyboardFunc(key);
	glutReshapeFunc(reshape);
        glutIdleFunc(run);

        /* Init the GL state */
        glLineWidth(1.0);

}


/* Key press processing */
void key(unsigned char c, int x, int y)
{
	if(c == 27) exit(0);
};


/* Call back when the windows is re-sized */
void reshape(GLsizei width, GLsizei height) {
   // Compute aspect ratio of the new window
   if (height == 0) height = 1;                // To prevent divide by 0
   GLfloat aspect = (GLfloat)width / (GLfloat)height;

   // Set the viewport to cover the new window
   glViewport(0, 0, width, height);

   // Set the aspect ratio of the clipping area to match the viewport
   glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
   glLoadIdentity();             // Reset the projection matrix
   if (width >= height) {
      clipAreaXLeft   = -1.0 * aspect;
      clipAreaXRight  = 1.0 * aspect;
      clipAreaYBottom = -1.0;
      clipAreaYTop    = 1.0;
   } else {
      clipAreaXLeft   = -1.0;
      clipAreaXRight  = 1.0;
      clipAreaYBottom = -1.0 / aspect;
      clipAreaYTop    = 1.0 / aspect;
   }
   gluOrtho2D(clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop);
   glutPostRedisplay();
}


void display() {

        float maxPhi = -1000;
        float minPhi =  1000;
        float yg[my],phig[my];
        for (int i=0; i<my; i++) {
            maxPhi  = MAX(maxPhi,(float)phi[idx(jd,i,kd)]);
            minPhi  = MIN(minPhi,(float)phi[idx(jd,i,kd)]);
            yg[i]   = (float) y[i];
            phig[i] = (float) phi[idx(jd,i,kd)];
        } 

	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glPushMatrix();
	glTranslatef(-(yg[my-1]+yg[0])/2.0, -(maxPhi+minPhi)/2.0, 0.0);
	glColor3f(1.0, 1.0, 1.0);

        glBegin(GL_LINE_STRIP); 
          for(int i=0; i<my; i++) {
                  glVertex2f(yg[i],phig[i]);
          }
        glEnd();

	glPopMatrix();
	glutSwapBuffers();

	glFlush();  // Render now
}

#endif







