/**
  @file sub_oda.cpp
  @author Hari Sundar, hsundar@gmail.com
 **/

#include "sub_oda.h"

namespace ot {

//***************Constructor*****************//
subDA::subDA(DA* da, std::function<double ( double, double, double ) > fx_retain, double* gSize) {
  m_da = da;
  unsigned int lev;
  
  unsigned maxDepth = m_da->getMaxDepth() - 1;


  // std::cout << "subDA: mD= " << maxDepth << std::endl;
  auto inside = [](double d){ return d < 0.0; };

  double hx, hy, hz;
  std::array<double, 8> dist;
  Point pt;

  double xFac = gSize[0]/((double)(1<<(maxDepth)));
  double yFac = gSize[1]/((double)(1<<(maxDepth)));
  double zFac = gSize[2]/((double)(1<<(maxDepth)));

  unsigned int indices[8];

  // now process the DA to skip interior elements
  m_ucpSkipList.clear();
  m_ucpSkipList.resize(m_da->getGhostedElementSize(), 0);

  //std::cout << "ghosted element size: " << m_ucpSkipList.size() << ", " << m_da->getNodeSize() << std::endl;

  m_ucpSkipNodeList.clear();
  m_ucpSkipNodeList.resize(m_da->getGhostedNodeSize(), 1);

  // m_uiNodeSize + m_uiBoundaryNodeSize + m_uiPreGhostNodeSize + m_uiPreGhostBoundaryNodeSize + m_uiPostGhostNodeSize

  // std::cout << "ghosted node Size: " << m_ucpSkipNodeList.size() << std::endl;
      
  for ( m_da->init<ot::DA_FLAGS::ALL>(); 
        m_da->curr() < m_da->end<ot::DA_FLAGS::ALL>(); 
        m_da->next<ot::DA_FLAGS::ALL>() ) {

          lev = m_da->getLevel(m_da->curr());
          hx = xFac*(1<<(maxDepth - lev));
          hy = yFac*(1<<(maxDepth - lev));
          hz = zFac*(1<<(maxDepth - lev));

          pt = m_da->getCurrentOffset();

          m_da->getNodeIndices(indices);


          dist[0] = fx_retain(pt.x()*xFac, pt.y()*yFac, pt.z()*zFac);
          dist[1] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac, pt.z()*zFac);
          dist[2] = fx_retain(pt.x()*xFac, pt.y()*yFac+hy, pt.z()*zFac);
          dist[3] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac+hy, pt.z()*zFac);

          dist[4] = fx_retain(pt.x()*xFac, pt.y()*yFac, pt.z()*zFac+hz);
          dist[5] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac, pt.z()*zFac+hz);
          dist[6] = fx_retain(pt.x()*xFac, pt.y()*yFac+hy, pt.z()*zFac+hz);
          dist[7] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac+hy, pt.z()*zFac +hz);

          /*
          for (auto q: dist)
            std::cout << q << ", ";
          std::cout << std::endl;
          */

          if ( std::all_of( dist.begin(), dist.end(), inside ) ) {
            // element to skip.
            // std::cout << "subDA: skip element" << std::endl;
            // std::cout << "s" << da->curr() << ", ";
            m_ucpSkipList[m_da->curr()] = 1;
          }
          else {
            // touch nodes ....
            for(int k = 0; k < 8; k++) {
              if ( indices[k] < m_ucpSkipList.size() )
                m_ucpSkipNodeList[indices[k]] = 0;
              // else
              //  std::cout << "skipList node index out of bound: " << indices[k] << std::endl;
              
            }
          }

        } // for 
   // std::cout << std::endl;
  // update counts ...

} // subDA constructor.

}; // namespace ot