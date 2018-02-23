/**
  @file sub_oda.tcc
  @author Hari Sundar, hsundar@gmail.com
 **/

namespace ot {

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // 
  // Implementation...

  //Next functions...
  template<ot::DA_FLAGS::loopType type>
    inline unsigned int subDA::next() {
      unsigned int current = m_da->next<type>();
      while ( m_ucpSkipList[current] ) {
        current = m_da->next<type>();
      } 
      return current;
    }//end function

  //Init functions...
  template<ot::DA_FLAGS::loopType type>	
    inline void subDA::init() {
      m_da->init<type>();
      if ( m_ucpSkipList[ m_da->curr() ] )
        m_da->next<type>();
    }//end function


  //End functions...
  template<ot::DA_FLAGS::loopType type>
    inline unsigned int subDA::end() {
      return m_da->end<type>();
    }

}; // namespace ot